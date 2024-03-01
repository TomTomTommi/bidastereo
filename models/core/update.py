# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead3D(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead3D, self).__init__()
        self.conv1 = nn.Conv3d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv3d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SKSepConvGRU3D(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(SKSepConvGRU3D, self).__init__()
        self.convz1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convr1 = nn.Sequential(
            nn.Conv3d(hidden_dim+input_dim, hidden_dim, (1, 1, 15), padding=(0, 0, 7)),
            nn.GELU(),
            nn.Conv3d(hidden_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)),
        )
        self.convq1 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 1, 5), padding=(0, 0, 2)
        )

        self.convz2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convr2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )
        self.convq2 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (1, 5, 1), padding=(0, 2, 0)
        )

        self.convz3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convr3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )
        self.convq3 = nn.Conv3d(
            hidden_dim + input_dim, hidden_dim, (5, 1, 1), padding=(2, 0, 0)
        )

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # time
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz3(hx))
        r = torch.sigmoid(self.convr3(hx))
        q = torch.tanh(self.convq3(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h


class MultiMotionEncoder(nn.Module):
    def __init__(self, cor_planes):
        super(MultiMotionEncoder, self).__init__()
        self.cor_planes = cor_planes
        self.convc1 = nn.Conv2d(cor_planes // 3, 96, 1, padding=0)
        self.convc2 = nn.Conv2d(288, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192 + 48 * 3, 128 - 2 + 48, 3, padding=1)

        self.init_hidden_state = nn.Parameter(torch.randn(1, 1, 48, 1, 1))

    def flow_warp(self, x,
                  flow,
                  interpolation='bilinear',
                  padding_mode='zeros',
                  align_corners=True):
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
            mode=interpolation,
            padding_mode=padding_mode,
            align_corners=align_corners)
        return output

    def forward(self, motion_hidden_state, flow_forward, flow_backward, flow, corr, t):

        BN, _, H, W = flow.shape
        bs = BN // t
        if motion_hidden_state is None:
            motion_hidden_state = self.init_hidden_state.repeat(bs, t, 1, H, W)
        else:
            motion_hidden_state = motion_hidden_state.reshape(bs, t, -1, H, W)

        backward_motion_hidden_state = rearrange(motion_hidden_state[:, 1:, ...], "b t c h w -> (b t) c h w")
        backward_motion_hidden_state = self.flow_warp(backward_motion_hidden_state, flow_backward)
        backward_motion_hidden_state = rearrange(backward_motion_hidden_state, "(b t) c h w -> b t c h w", b=bs, t=t-1)
        backward_motion_hidden_state = torch.cat((backward_motion_hidden_state,motion_hidden_state[:, -1:, ...]), dim = 1)

        forward_motion_hidden_state = rearrange(motion_hidden_state[:, :t-1, ...], "b t c h w -> (b t) c h w")
        forward_motion_hidden_state = self.flow_warp(forward_motion_hidden_state, flow_forward)
        forward_motion_hidden_state = rearrange(forward_motion_hidden_state, "(b t) c h w -> b t c h w", b=bs, t=t-1)
        forward_motion_hidden_state = torch.cat((motion_hidden_state[:, :1, ...], forward_motion_hidden_state), dim=1)

        corr0, corr1, corr2 = torch.split(corr, [self.cor_planes // 3, self.cor_planes // 3, self.cor_planes // 3], dim=1)

        cor = F.gelu(torch.cat([self.convc1(corr0), self.convc1(corr1), self.convc1(corr2)], dim=1))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # cor_flo = torch.cat([cor, flo], dim=1)
        cor_flo = torch.cat([cor, flo, forward_motion_hidden_state.reshape(BN, -1, H, W), backward_motion_hidden_state.reshape(BN, -1, H, W),
                             motion_hidden_state.reshape(BN, -1, H, W)], dim=1)
        out = F.relu(self.conv(cor_flo))
        out, motion_hidden_state = torch.split(out, [126, 48], dim=1)
        return torch.cat([out, flow], dim=1), motion_hidden_state


class MultiSequenceUpdateBlock3D(nn.Module):
    def __init__(self, hidden_dim, cor_planes, mask_size=8):
        super(MultiSequenceUpdateBlock3D, self).__init__()

        self.encoder = MultiMotionEncoder(cor_planes)
        self.gru = SKSepConvGRU3D(hidden_dim=hidden_dim, input_dim=128 + hidden_dim)
        self.flow_head = FlowHead3D(hidden_dim, hidden_dim=256)
        self.mask = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim + 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim + 128, (mask_size ** 2) * 9, 1, padding=0),
        )

    def forward(self, net, inp, corrs, flows, motion_hidden_state, flow_forward, flow_backward, t):
        motion_features, motion_hidden_state = self.encoder(motion_hidden_state, flow_forward, flow_backward, flows, corrs, t=t)
        inp_tensor = torch.cat([inp, motion_features], dim=1)

        net = rearrange(net, "(b t) c h w -> b c t h w", t=t)
        inp_tensor = rearrange(inp_tensor, "(b t) c h w -> b c t h w", t=t)

        net = self.gru(net, inp_tensor)

        delta_flow = self.flow_head(net)

        # scale mask to balance gradients
        net = rearrange(net, " b c t h w -> (b t) c h w")
        mask = 0.25 * self.mask(net)

        delta_flow = rearrange(delta_flow, " b c t h w -> (b t) c h w")
        return net, mask, delta_flow, motion_hidden_state



