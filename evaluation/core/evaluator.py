# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import cv2
from collections import defaultdict
import torch.nn.functional as F
import torch
from tqdm import tqdm
from omegaconf import DictConfig
from pytorch3d.implicitron.tools.config import Configurable
from bidastereo.evaluation.utils.eval_utils import depth2disparity_scale, eval_batch
from bidastereo.evaluation.utils.utils import (
    PerceptionPrediction,
    pretty_print_perception_metrics,
    visualize_batch,
)


class Evaluator(Configurable):
    """
    A class defining the DynamicStereo evaluator.

    Args:
        eps: Threshold for converting disparity to depth.
    """

    eps = 1e-5

    def setup_visualization(self, cfg: DictConfig) -> None:
        # Visualization
        self.visualize_interval = cfg.visualize_interval
        self.exp_dir = cfg.exp_dir
        if self.visualize_interval > 0:
            self.visualize_dir = os.path.join(cfg.exp_dir, "visualisations")

    @torch.no_grad()
    def evaluate_sequence(
        self,
        model,
        test_dataloader: torch.utils.data.DataLoader,
        is_real_data: bool = False,
        step=None,
        writer=None,
        train_mode=False,
        interp_shape=None,
        exp_dir=None,
    ):
        model.eval()
        per_batch_eval_results = []

        if self.visualize_interval > 0:
            os.makedirs(self.visualize_dir, exist_ok=True)

        for batch_idx, sequence in enumerate(tqdm(test_dataloader)):
            batch_dict = defaultdict(list)
            batch_dict["stereo_video"] = sequence["img"]
            if not is_real_data:
                batch_dict["disparity"] = sequence["disp"][:, 0].abs()
                batch_dict["disparity_mask"] = sequence["valid_disp"][:, :1]

                if "mask" in sequence:
                    batch_dict["fg_mask"] = sequence["mask"][:, :1]
                else:
                    batch_dict["fg_mask"] = torch.ones_like(
                        batch_dict["disparity_mask"]
                    )

            elif interp_shape is not None:
                left_video = batch_dict["stereo_video"][:, 0]
                left_video = F.interpolate(
                    left_video, tuple(interp_shape), mode="bilinear"
                )
                right_video = batch_dict["stereo_video"][:, 1]
                right_video = F.interpolate(
                    right_video, tuple(interp_shape), mode="bilinear"
                )
                batch_dict["stereo_video"] = torch.stack([left_video, right_video], 1)

            if train_mode:
                predictions = model.forward_batch_test(batch_dict)
            else:
                predictions = model(batch_dict)

            assert "disparity" in predictions
            predictions["disparity"] = predictions["disparity"][:, :1].clone().cpu()

            if not is_real_data:
                predictions["disparity"] = predictions["disparity"] * (
                    batch_dict["disparity_mask"].round()
                )

                batch_eval_result, seq_length = eval_batch(batch_dict, predictions)

                per_batch_eval_results.append((batch_eval_result, seq_length))
                pretty_print_perception_metrics(batch_eval_result)

            if (self.visualize_interval > 0) and (
                batch_idx % self.visualize_interval == 0
            ):
                perception_prediction = PerceptionPrediction()

                pred_disp = predictions["disparity"]
                pred_disp[pred_disp < self.eps] = self.eps

                scale = depth2disparity_scale(
                    sequence["viewpoint"][0][0],
                    sequence["viewpoint"][0][1],
                    torch.tensor([pred_disp.shape[2], pred_disp.shape[3]])[None],
                )

                perception_prediction.depth_map = (scale / pred_disp).cuda()
                perspective_cameras = []
                for cam in sequence["viewpoint"]:
                    perspective_cameras.append(cam[0])

                perception_prediction.perspective_cameras = perspective_cameras

                if "stereo_original_video" in batch_dict:
                    batch_dict["stereo_video"] = batch_dict[
                        "stereo_original_video"
                    ].clone()

                for k, v in batch_dict.items():
                    if isinstance(v, torch.Tensor):
                        batch_dict[k] = v.cuda()

                visualize_batch(
                    batch_dict,
                    perception_prediction,
                    output_dir=self.visualize_dir,
                    sequence_name=sequence["metadata"][0][0][0],
                    step=step,
                    writer=writer,
                )
                filename = os.path.join(self.visualize_dir, sequence["metadata"][0][0][0])
                if not os.path.isdir(filename):
                    os.mkdir(filename)
                depth_list = perception_prediction.depth_map.data.cpu().numpy()
                depth_list = ((depth_list - depth_list.min()) / (depth_list.max() - depth_list.min())* 255).astype(np.uint8)
                print("depth_list:", depth_list.shape)
                video_depth = cv2.VideoWriter(
                    f"{filename}_depth.mp4",
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=10,
                    frameSize=(
                    batch_dict["stereo_video"][:, 0][0].shape[2], batch_dict["stereo_video"][:, 0][0].shape[1]),
                    isColor=True,
                )
                for i in range(depth_list.shape[0]):
                    depth = depth_list[i]
                    depth = depth.transpose(1,2,0)
                    depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
                    filename_temp = filename + '/depth_' + str(i).zfill(3) + '.png'
                    cv2.imwrite(filename_temp, depth_vis)
                    video_depth.write(depth_vis)
                video_depth.release()

        return per_batch_eval_results
