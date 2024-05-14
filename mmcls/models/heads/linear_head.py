# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS, build_loss
from .cls_head import ClsHead


@HEADS.register_module()
class LinearClsHead(ClsHead):
    """Linear classifier head.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        init_cfg (dict | optional): The extra init config of layers.
            Defaults to use dict(type='Normal', layer='Linear', std=0.01).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(LinearClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)

        self.in_channels = in_channels
        self.num_classes = num_classes

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        self.fc = nn.Linear(self.in_channels, self.num_classes)

    def pre_logits(self, x):
        if isinstance(x, tuple):
            x = x[-1]
        return x
    def simple_test(self, x, softmax=True, post_process=True):
        """Inference without augmentation.

        Args:
            x (tuple[Tensor]): The input features.
                Multi-stage inputs are acceptable but only the last stage will
                be used to classify. The shape of every item should be
                ``(num_samples, in_channels)``.
            softmax (bool): Whether to softmax the classification score.
            post_process (bool): Whether to do post processing the
                inference results. It will convert the output to a list.

        Returns:
            Tensor | list: The inference results.

                - If no post processing, the output is a tensor with shape
                  ``(num_samples, num_classes)``.
                - If post processing, the output is a multi-dimentional list of
                  float and the dimensions are ``(num_samples, num_classes)``.
        """
        x = self.pre_logits(x)
        cls_score = self.fc(x)

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        if post_process:
            return self.post_process(pred)
        else:
            return pred

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        cls_score = self.fc(x)
    
        losses = self.loss(cls_score, gt_label, **kwargs)
        return losses


@HEADS.register_module()
class IdentityClsHead(ClsHead):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_train(self, x, gt_label, **kwargs):
        x = self.pre_logits(x)
        # cls_score = self.fc(x)
        losses = self.loss(x, gt_label, **kwargs)
        return losses

@HEADS.register_module()
class IdentityClsHead_hybridL(ClsHead):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def loss_hybridL(self, x, gt_label,k,x0,img, **kwargs):
        num_samples = len(x)
        losses = dict()
        # compute loss
        loss = self.compute_loss(
            x, gt_label,k,x0,img,avg_factor=num_samples, **kwargs)
        if self.cal_acc:
            # compute accuracy
            acc = self.compute_accuracy(cls_score, gt_label)
            assert len(acc) == len(self.topk)
            losses['accuracy'] = {
                f'top-{k}': a
                for k, a in zip(self.topk, acc)
            }
        losses['loss'] = loss
        return losses
    def forward_train(self, x, gt_label,k,x0,img,**kwargs):
       # print(k.shape)
       # print(k)
        x = self.pre_logits(x)
        # cls_score = self.fc(x)
        losses = self.loss_hybridL(x, gt_label,k,x0,img, **kwargs)
        return losses


# @HEADS.register_module()
# class FaceLinearClsHead(LinearClsHead):
#     def __init__(self,
#                  num_classes,
#                  in_channels,
#                  init_cfg=dict(type='Normal', layer='Linear', std=0.01),
#                  loss_face=None,
#                  *args,
#                  **kwargs):
#         super().__init__(
#             num_classes,
#             in_channels,
#             init_cfg,
#             *args,
#             **kwargs)
#         self.face_loss = build_loss(loss_face)
#
#     def forward_train(self, x, gt_label, **kwargs):
#         x = self.pre_logits(x)
#         x = self.face_loss(x, gt_label)
#         cls_score = self.fc(x)
#         losses = self.loss(cls_score, gt_label, **kwargs)
#         return losses
