# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from .segment_anything_fast.modeling import Sam

from typing import Optional, Tuple
from torch.nn import functional as F

from .segment_anything_fast.utils.transforms import ResizeLongestSide
from .segment_anything_fast import sam_model_registry

import logging
from typing import List, Optional

import torch.nn as nn
import torch.nn.functional as F
from mmengine.logging import print_log
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .encoder_decoder import EncoderDecoder

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


@MODELS.register_module()
class BuildSAM(EncoderDecoder):
    def __init__(
        self,
        backbone: ConfigType = None,
        decode_head: ConfigType = None,
        neck: OptConfigType = None,
        auxiliary_head: OptConfigType = None,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        pretrained: Optional[str] = None,
        init_cfg: OptMultiConfig = None,
    ):
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__(backbone=backbone, decode_head=decode_head, data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        sam_checkpoint = "/home/ps/.cache/torch/hub/checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"
        self.model = sam_model_registry[model_type](checkpoint=sam_checkpoint)

        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)
        self.reset_image()

        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def encode_decode(self, inputs: Tensor,
        batch_img_metas: List[dict],
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = True,
    ) -> Tensor:
        losses = dict()
        with torch.no_grad():
            self.features = self.model.image_encoder(inputs)
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        point_coords = batch_img_metas[0]['point_coords']
        point_labels = batch_img_metas[0]['point_labels']

        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        sparse_embeddings, dense_embeddings, multimask_output = self.prompt_encoder(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output
        )
        masks, iou_predictions, low_res_masks = self.mask_decoder(sparse_embeddings, dense_embeddings, multimask_output, return_logits)
        # logits = masks[:, torch.argmax(iou_predictions[0]), :, :].unsqueeze(1)
        logits = masks

        return logits

    def loss(self, inputs: Tensor, 
        data_samples: SampleList,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
        return_logits: bool = True,
    ) -> dict:
        losses = dict()
        with torch.no_grad():
            self.features = self.model.image_encoder(inputs)
        # Transform input prompts
        coords_torch, labels_torch, box_torch, mask_input_torch = None, None, None, None
        point_coords = data_samples[0].point_coords
        point_labels = data_samples[0].point_labels

        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = self.transform.apply_coords(point_coords, self.original_size)
            coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
            labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        if box is not None:
            box = self.transform.apply_boxes(box, self.original_size)
            box_torch = torch.as_tensor(box, dtype=torch.float, device=self.device)
            box_torch = box_torch[None, :]
        if mask_input is not None:
            mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
            mask_input_torch = mask_input_torch[None, :, :, :]

        sparse_embeddings, dense_embeddings, multimask_output = self.prompt_encoder(
            coords_torch,
            labels_torch,
            box_torch,
            mask_input_torch,
            multimask_output
        )
        masks, iou_predictions, low_res_masks = self.mask_decoder(sparse_embeddings, dense_embeddings, multimask_output, return_logits)
        # logits = masks[:, torch.argmax(iou_predictions[0]), :, :].unsqueeze(1)
        logits = masks
        loss_decode = self._decode_head_forward_train(logits, data_samples)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train([logits], data_samples)
            losses.update(loss_aux)
        return losses

    @torch.no_grad()
    def prompt_encoder(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using ResizeLongestSide.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """
        if point_coords is not None:
            points = (point_coords, point_labels)
        else:
            points = None

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        return sparse_embeddings, dense_embeddings, multimask_output

    def mask_decoder(self, sparse_embeddings, dense_embeddings, multimask_output, return_logits=True):
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=self.features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        if low_res_masks.is_nested:
            masks = []
            for lrm, input_size, original_size in zip(low_res_masks.unbind(), self.input_sizes, self.original_sizes, strict=True):
                # Upscale the masks to the original image resolution
                m = self.model.postprocess_masks(lrm, input_size, original_size)
                masks.append(m)
            masks = torch.nested.nested_tensor(masks, layout=torch.strided)
        else:
            # Upscale the masks to the original image resolution
            masks = self.model.postprocess_masks(low_res_masks, self.input_size, self.original_size)

        if not return_logits:
            masks = masks > self.model.mask_threshold

        return masks, iou_predictions, low_res_masks

    def get_image_embedding(self) -> torch.Tensor:
        """
        Returns the image embeddings for the currently set image, with
        shape 1xCxHxW, where C is the embedding dimension and (H,W) are
        the embedding spatial dimension of SAM (typically C=256, H=W=64).
        """
        if not self.is_image_set:
            raise RuntimeError(
                "An image must be set with .set_image(...) to generate an embedding."
            )
        assert self.features is not None, "Features must exist if an image has been set."
        return self.features

    @property
    def device(self) -> torch.device:
        return self.model.device

    def reset_image(self) -> None:
        """Resets the currently set image."""
        self.is_image_set = False
        self.features = None
        self.orig_h = None
        self.orig_w = None
        self.input_h = None
        self.input_w = None
        self.original_size = (1024, 1024)
        self.input_size = (1024, 1024)
