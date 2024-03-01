from typing import Dict, List, ClassVar
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import importlib
import sys

from bidastereo.models.core.update import (
    MultiSequenceUpdateBlock3D,
)
from bidastereo.models.core.extractor import BasicEncoder, ResidualBlock
from bidastereo.models.core.corr import TFCL

from bidastereo.models.core.utils.utils import InputPadder, interp
from bidastereo.models.raft_model import RAFTModel

autocast = torch.cuda.amp.autocast


class BiDAStereo(nn.Module):
    def __init__(self, mixed_precision = False):
        super(BiDAStereo, self).__init__()

        self.mixed_precision = mixed_precision

        self.hidden_dim = 128
        self.context_dim = 128
        self.dropout = 0

        self.raft = RAFTModel()

        # feature network and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
        self.update_block = MultiSequenceUpdateBlock3D(hidden_dim=self.hidden_dim, cor_planes=3*9, mask_size=4)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"time_embed"}

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def convex_upsample(self, flow, mask, rate=4):
        """ Upsample flow field [H/rate, W/rate, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, rate, rate, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, rate * H, rate * W)

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        flow_u = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow_v = torch.zeros([N, 1, H, W], dtype=torch.float)
        flow = torch.cat([flow_u, flow_v], dim=1).to(fmap.device)
        return flow

    def forward_batch_test(
        self, batch_dict: Dict, kernel_size: int = 14, iters: int = 20
    ):
        disp_preds = []
        predictions = defaultdict(list)
        video = batch_dict["stereo_video"]
        num_ims = len(video)
        print("video", video.shape)
        if kernel_size >= num_ims:
            left_ims = video[:, 0]
            right_ims = video[:, 1]
            padder = InputPadder(left_ims.shape, divis_by=32)
            left_ims, right_ims = padder.pad(left_ims, right_ims)

            with autocast(enabled=self.mixed_precision):
                disparities_forw = self.forward(
                    left_ims[None].cuda(),
                    right_ims[None].cuda(),
                    iters=iters,
                    test_mode=True,
                )
            disparities_forw = padder.unpad(disparities_forw[:, 0])[:, None].cpu()
            disp_preds.append(disparities_forw)

            predictions["disparity"] = (torch.cat(disp_preds).squeeze(1).abs())[:, :1]
            print(predictions["disparity"].shape)

            return predictions

        else:
            stride = kernel_size // 2
            EXTEND_VIDEO = False
            if (num_ims - kernel_size) % stride == 1:
                EXTEND_VIDEO = True
                video = torch.cat((video, video[num_ims-1:num_ims, :]), dim=0)
                num_ims = num_ims + 1
                print("extended video", video.shape)
            for i in range(0, num_ims, stride):
                left_ims = video[i : min(i + kernel_size, num_ims), 0]
                padder = InputPadder(left_ims.shape, divis_by=32)

                right_ims = video[i : min(i + kernel_size, num_ims), 1]
                left_ims, right_ims = padder.pad(left_ims, right_ims)

                with autocast(enabled=self.mixed_precision):
                    disparities_forw = self.forward(
                        left_ims[None].cuda(),
                        right_ims[None].cuda(),
                        iters=iters,
                        test_mode=True,
                    )

                disparities_forw = padder.unpad(disparities_forw[:, 0])[:, None].cpu()

                if len(disp_preds) > 0 and len(disparities_forw) >= stride:

                    if len(disparities_forw) < kernel_size:
                        disp_preds.append(disparities_forw[stride // 2 :])
                    else:
                        disp_preds.append(disparities_forw[stride // 2 : -stride // 2])

                elif len(disp_preds) == 0:
                    disp_preds.append(disparities_forw[: -stride // 2])

            predictions["disparity"] = (torch.cat(disp_preds).squeeze(1).abs())[:, :1]
            if EXTEND_VIDEO:
                predictions["disparity"] = predictions["disparity"][:num_ims-1,:,:,:]
            print(predictions["disparity"].shape)

            return predictions

    def compute_flow(self, seq):
        n, t, c, h, w = seq.size()
        flows_forward = []
        flows_backward = []
        for i in range(t-1):
            # i-th flow_backward denotes seq[i+1] towards seq[i]
            flow_backward = self.raft(seq[:,i], seq[:,i+1])
            # i-th flow_forward denotes seq[i] towards seq[i+1]
            flow_forward = self.raft(seq[:,i+1], seq[:,i])
            flows_backward.append(flow_backward)
            flows_forward.append(flow_forward)

        return flows_forward, flows_backward

    def flow_warp(self, x, flow):
        if flow.size(3) != 2:  # [B, H, W, 2]
            flow = flow.permute(0, 2, 3, 1)
        if x.size()[-2:] != flow.size()[1:3]:
            raise ValueError(f'The spatial sizes of input ({x.size()[-2:]}) and '
                             f'flow ({flow.size()[1:3]}) are not the same.')
        _, _, h, w = x.size()
        # create mesh grid
        grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
        grid = torch.stack((grid_x, grid_y), 2).type_as(x)  # (h, w, 2)
        grid.requires_grad = False

        grid_flow = grid + flow
        # scale grid_flow to [-1,1]
        grid_flow_x = 2.0 * grid_flow[:, :, :, 0] / max(w - 1, 1) - 1.0
        grid_flow_y = 2.0 * grid_flow[:, :, :, 1] / max(h - 1, 1) - 1.0
        grid_flow = torch.stack((grid_flow_x, grid_flow_y), dim=3)
        output = F.grid_sample(
            x,
            grid_flow,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True)
        return output

    def forward(self, seq1, seq2, flow_init=None, iters=10, test_mode=False):
        b, T, *_ = seq1.shape

        # compute optical flow
        flows_forward2, flows_backward2 = self.compute_flow(seq2)

        flow_forward2 = torch.stack(flows_forward2, dim=1)
        flow_backward2 = torch.stack(flows_backward2, dim=1)
        flow_forward2 = rearrange(flow_forward2, "b t c h w -> (b t) c h w")
        flow_backward2 = rearrange(flow_backward2, "b t c h w -> (b t) c h w")
        s_flow_forward2 = 1 / 2 * F.interpolate(flow_forward2,
                                                size=(flow_forward2.shape[2] // 2, flow_forward2.shape[3] // 2),
                                                mode='bilinear',
                                                align_corners=True)
        s_flow_backward2 = 1 / 2 * F.interpolate(flow_backward2,
                                                size=(flow_backward2.shape[2] // 2, flow_backward2.shape[3] // 2),
                                                mode='bilinear',
                                                align_corners=True)
        ss_flow_forward2 = 1 / 4 * F.interpolate(flow_forward2,
                                                size=(flow_forward2.shape[2] // 4, flow_forward2.shape[3] // 4),
                                                mode='bilinear',
                                                align_corners=True)
        ss_flow_backward2 = 1 / 4 * F.interpolate(flow_backward2,
                                                size=(flow_backward2.shape[2] // 4, flow_backward2.shape[3] // 4),
                                                mode='bilinear',
                                                align_corners=True)

        seq1 = rearrange(seq1, "b t c h w -> (b t) c h w")
        seq2 = rearrange(seq2, "b t c h w -> (b t) c h w")
        seq1 = 2 * (seq1 / 255.0) - 1.0
        seq2 = 2 * (seq2 / 255.0) - 1.0
        seq1 = seq1.contiguous()
        seq2 = seq2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # feature network
        with autocast(enabled=self.mixed_precision):
            seqmap1, seqmap2 = self.fnet([seq1, seq2])  # 256 * H/4 * W/4

        seqmap2 = rearrange(seqmap2, "(b t) c h w -> b t c h w", b=b, t=T)
        seqmap2 = seqmap2.float()

        sequence_backward = rearrange(seqmap2[:, 1:, ...], "b t c h w -> (b t) c h w")
        feat_prop2_backward = self.flow_warp(sequence_backward, flow_backward2)
        feat_prop2_backward = rearrange(feat_prop2_backward, "(b t) c h w -> b t c h w", b=b, t=T-1)
        output2_backward = torch.cat((feat_prop2_backward, seqmap2[:,-1:]), dim = 1)

        sequence_forward = rearrange(seqmap2[:, :T-1, ...], "b t c h w -> (b t) c h w")
        feat_prop2_forward = self.flow_warp(sequence_forward, flow_forward2)
        feat_prop2_forward = rearrange(feat_prop2_forward, "(b t) c h w -> b t c h w", b=b, t=T-1)
        output2_forward = torch.cat((seqmap2[:, :1], feat_prop2_forward), dim=1)

        fmap1 = seqmap1
        fmap2 = torch.cat((seqmap2, output2_forward, output2_backward), dim=1)
        fmap2 = rearrange(fmap2, "b t c h w -> (b t) c h w")

        with autocast(enabled=self.mixed_precision):
            # 1/4 -> 1/8
            # feature
            s_fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            s_fmap2 = F.avg_pool2d(fmap2, 2, stride=2)

            # context
            net, inp = torch.split(fmap1, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            s_net = F.avg_pool2d(net, 2, stride=2)
            s_inp = F.avg_pool2d(inp, 2, stride=2)

            # 1/4 -> 1/16
            # feature
            ss_fmap1 = F.avg_pool2d(fmap1, 4, stride=4)
            ss_fmap2 = F.avg_pool2d(fmap2, 4, stride=4)

            # context
            ss_net = F.avg_pool2d(net, 4, stride=4)
            ss_inp = F.avg_pool2d(inp, 4, stride=4)

        # Triple Frame Correlation Layer
        corr_fn = TFCL(fmap1, fmap2)
        s_corr_fn = TFCL(s_fmap1, s_fmap2)
        ss_corr_fn = TFCL(ss_fmap1, ss_fmap2)

        # cascaded refinement (1/16 + 1/8 + 1/4)
        flow_predictions = []
        flow = None
        flow_up = None

        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = scale * interp(flow_init, size=(fmap1.shape[2], fmap1.shape[3]))
        else:
            # init flow
            ss_flow = self.zero_init(ss_fmap1)  # 256 * H/16 * W/16 --> 2 * H/16 * W/16
            ss_motion_hidden_state = None

            # 1/16
            for itr in range(iters//2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                ss_flow = ss_flow.detach()
                out_corrs = ss_corr_fn(ss_flow, None, small_patch=small_patch)

                with autocast(enabled=self.mixed_precision):
                    ss_net, up_mask, delta_flow, ss_motion_hidden_state = self.update_block(ss_net, ss_inp, out_corrs, ss_flow, ss_motion_hidden_state, ss_flow_forward2, ss_flow_backward2, t=T)

                ss_flow = ss_flow + delta_flow
                flow = self.convex_upsample(ss_flow, up_mask, rate=4)
                flow_up = 4 * F.interpolate(flow, size=(4 * flow.shape[2], 4 * flow.shape[3]), mode='bilinear', align_corners=True)
                flow_predictions.append(flow_up[:, :1])

            scale = s_fmap1.shape[2] / flow.shape[2]
            s_flow = scale * interp(flow, size=(s_fmap1.shape[2], s_fmap1.shape[3]))
            s_motion_hidden_state = F.interpolate(ss_motion_hidden_state, size=(2 * ss_motion_hidden_state.shape[2], 2 * ss_motion_hidden_state.shape[3]), mode='bilinear',
                                        align_corners=True)

            # 1/8
            for itr in range(iters//2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                s_flow = s_flow.detach()
                out_corrs = s_corr_fn(s_flow, None, small_patch=small_patch)

                with autocast(enabled=self.mixed_precision):
                    s_net, up_mask, delta_flow, s_motion_hidden_state = self.update_block(s_net, s_inp, out_corrs, s_flow, s_motion_hidden_state, s_flow_forward2, s_flow_backward2, t=T)

                s_flow = s_flow + delta_flow
                flow = self.convex_upsample(s_flow, up_mask, rate=4)
                flow_up = 2 * F.interpolate(flow, size=(2 * flow.shape[2], 2 * flow.shape[3]), mode='bilinear', align_corners=True)
                flow_predictions.append(flow_up[:, :1])

            scale = fmap1.shape[2] / flow.shape[2]
            flow = scale * interp(flow, size=(fmap1.shape[2], fmap1.shape[3]))

        motion_hidden_state = F.interpolate(s_motion_hidden_state, size=(2 * s_motion_hidden_state.shape[2],
                                                                         2 * s_motion_hidden_state.shape[3]), mode='bilinear',align_corners=True)  # 2 * H/2 * W/2
        # 1/4
        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch)

            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow, motion_hidden_state = self.update_block(net, inp, out_corrs, flow, motion_hidden_state, flow_forward2, flow_backward2, t=T)

            flow = flow + delta_flow
            flow_up = self.convex_upsample(flow, up_mask, rate=4)
            flow_predictions.append(flow_up[:, :1])

        predictions = torch.stack(flow_predictions)
        predictions = rearrange(predictions, "d (b t) c h w -> d t b c h w", b=b, t=T)
        flow_up = predictions[-1]

        if test_mode:
            return flow_up

        return predictions