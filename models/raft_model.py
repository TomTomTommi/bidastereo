# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from typing import ClassVar
import torch.nn.functional as F

from pytorch3d.implicitron.tools.config import Configurable
import torch
import importlib
import sys

sys.path.append("third_party/RAFT")
raft = importlib.import_module(
    "bidastereo.third_party.RAFT.core.raft"
)


class RAFTModel(Configurable, torch.nn.Module):
    MODEL_CONFIG_NAME: ClassVar[str] = "RAFTModel"

    def __post_init__(self):
        super().__init__()
        self.model_weights: str = "./third_party/RAFT/models/raft-sintel.pth"

        model_args = SimpleNamespace(
            mixed_precision=False,
            small=False,
            dropout=0.0,
        )
        self.args = model_args
        self.model = raft.RAFT(model_args).cuda()

        state_dict = torch.load(self.model_weights)
        weight_dict = {}
        for k,v in state_dict.items():
            temp_k = k.replace('module.', '') if 'module' in k else k
            weight_dict[temp_k] = v
        self.model.load_state_dict(weight_dict)


    def forward(self, image1, image2):
        flow, flow_up = self.model(image1, image2, iters=10, test_mode=True)

        return 0.25 * F.interpolate(flow_up, size=(flow_up.shape[2] // 4, flow_up.shape[3] // 4), mode="bilinear",
        align_corners=True)
