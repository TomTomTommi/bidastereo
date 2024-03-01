# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    if H > 1:
        ygrid = 2 * ygrid/(H - 1) - 1
    img = img.contiguous()
    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous()
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device), indexing="ij"
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class TFCL:
    """
    Implementation of Triple-frame Correlation Layer (TFCL).
    """
    def __init__(self, fmap1, fmap2):
        self.fmap1 = fmap1
        self.fmap2 = fmap2
        self.coords = coords_grid(fmap1.shape[0], fmap1.shape[2], fmap1.shape[3], fmap1.device)

    def __call__(self, flow, extra_offset, small_patch=False):

        corr = self.correlation(self.fmap1, self.fmap2, flow, small_patch)

        return corr

    def correlation(self, left_feature, right_feature, flow, small_patch):
        flow[:, 1:] = 0
        coords = self.coords + flow
        coords = coords.permute(0, 2, 3, 1)
        coords = coords.repeat(3,1,1,1)
        right_feature = bilinear_sampler(right_feature, coords)

        if small_patch:
            psize_list = [(3, 3), (3, 3), (3, 3), (3, 3)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]
        else:
            psize_list = [(1, 9), (1, 9), (1, 9), (1, 9)]
            dilate_list = [(1, 1), (1, 1), (1, 1), (1, 1)]

        N, C, H, W = right_feature.size()
        rights = torch.split(right_feature, [N // 3] * 3, dim=0)
        corrs = []
        for i in range(3):
            corr = self.get_correlation(left_feature, rights[i], psize_list[i], dilate_list[i])
            corrs.append(corr)

        final_corr = torch.cat(corrs, dim=1)

        return final_corr

    def get_correlation(self, left_feature, right_feature, psize=(3, 3), dilate=(1, 1)):

        N, C, H, W = left_feature.size()

        di_y, di_x = dilate[0], dilate[1]
        pady, padx = psize[0] // 2 * di_y, psize[1] // 2 * di_x

        right_pad = F.pad(right_feature, [padx, padx, pady, pady], mode='replicate')

        corr_list = []
        for h in range(0, pady * 2 + 1, di_y):
            for w in range(0, padx * 2 + 1, di_x):
                right_crop = right_pad[:, :, h:h + H, w:w + W]
                assert right_crop.size() == left_feature.size()
                corr = (left_feature * right_crop).mean(dim=1, keepdim=True)
                corr_list.append(corr)

        corr_final = torch.cat(corr_list, dim=1)

        return corr_final