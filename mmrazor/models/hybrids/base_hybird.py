# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import HYBRIDS, build_posenet, build_classifier, build_optical
import torch.nn as nn
from mmcv.runner import BaseModule
from collections import OrderedDict
import torch
import torch.distributed as dist
# from mmdet.core import encode_mask_results
import torch.nn.functional as F
import random
import copy

@HYBRIDS.register_module()
class BaseHybrid(BaseModule):
    def __init__(self,
                 optical,
                 img_size=224,
                 remove_bg=False,
                 classifier=None,
                 posenet=None,
                 init_cfg=None
                 ):
        super().__init__(init_cfg=init_cfg)
        if optical is not None:
            self.optical = build_optical(optical)
            left = (img_size - self.optical.output_dim[1]) // 2
            right = img_size - self.optical.output_dim[1] - left
            top = (img_size - self.optical.output_dim[0]) // 2
            bottom = img_size - self.optical.output_dim[0] - top
            self.padding = (left, right, top, bottom)
        else:
            self.optical = None
        if classifier is not None:
            classifier['backbone']['img_size'] = img_size
            self.classifier = build_classifier(classifier)
        if posenet is not None:
            posenet['backbone']['img_size'] = img_size
            self.posenet = build_posenet(posenet)
        self.remove_bg = remove_bg

    def forward_optical(self, img):
        if len(img.shape) == 5:
            if self.remove_bg:
                img = img[:,1]
            else:
                img = img[:,0]
        img = img.squeeze(1)
        if self.optical is not None:
            img = self.optical(img)
            img = F.pad(img, self.padding, mode='constant', value=0)
        return img


    def forward_train(self, input_dict):
        """Forward computation during training.
        """
        losses_all = {}
        #for visualization both cls and pose
        if random.random() < 0.5:
            if hasattr(self, 'classifier'):
                data = input_dict['cls']

                img = self.forward_optical(data["img"])
                # copy data except img
                data_copy = data.copy()
                data_copy["img"] = img
                losses = self.classifier(**data_copy, return_loss=True)

                for key in losses.keys():
                    losses_all['cls_'+key] = losses[key]

          

            if hasattr(self, 'posenet'):
                data = input_dict['pose']
                img = self.forward_optical(data["img"])
                # copy data except img
                data_copy = data.copy()
                data_copy["img"] = img
                losses = self.posenet(**data_copy, return_loss=True)
                for key in losses.keys():
                    losses_all['pose_'+key] = losses[key]
        else:

            if hasattr(self, 'posenet'):
                data = input_dict['pose']
                img = self.forward_optical(data["img"])
                # copy data except img
                data_copy = data.copy()
                data_copy["img"] = img
                losses = self.posenet(**data_copy, return_loss=True)
                for key in losses.keys():
                    losses_all['pose_'+key] = losses[key]
            if hasattr(self, 'classifier'):
                data = input_dict['cls']

                img = self.forward_optical(data["img"])
                # copy data except img
                data_copy = data.copy()
                data_copy["img"] = img
                losses = self.classifier(**data_copy, return_loss=True)

                for key in losses.keys():
                    losses_all['cls_'+key] = losses[key]

      


       # print(losses)
        
        # losses_kernel_tv = self.tvloss(kernel)
        # losses_kernel_bin = self.binloss(kernel)
        # losses_kernel_inv = self.invloss(kernel)       
        # losses_all['kernel_tv_loss'] = losses_kernel_tv
        # losses_all['kernel_bin_loss'] = losses_kernel_bin
        # losses_all['kernel_inv_loss'] = losses_kernel_inv
        # print(losses_all)
       # print(losses_all)
        return losses_all

    def forward_test(self, input_dict):
        num_batch = 0
        for key in input_dict.keys():
            num_batch = max(num_batch, len(input_dict[key]['img']))

        results = [{}] * num_batch
        if 'cls' in input_dict.keys():
            data = input_dict['cls']
            img = data['img']
            if img.dim() == 5:
                b, n, _, _, _ = img.shape
                img = img.flatten(0, 1)
                img = self.forward_optical(img)
                _, c, h, w = img.shape
                img = img.reshape(b, n, c, h, w)
                data['img'] = img
            else:
                data['img'] = self.forward_optical(img)
            result = self.classifier(**data, return_loss=False)
            for i in range(len(result)):
                results[i]['cls_result'] = result[i]

        if 'pose' in input_dict.keys():
            data = input_dict['pose']
            data['img'] = self.forward_optical(data['img'])
            result = self.posenet(**data, return_loss=False)
            results[0]['pose_result'] = result

        return results

    def forward(self, input_dict, return_loss=True):
        if return_loss:
            return self.forward_train(input_dict)
        else:
            return self.forward_test(input_dict)

    def init_weights(self):

        if hasattr(self, 'classifier'):
            self.classifier.init_weights()


        if hasattr(self, 'posenet'):
            self.posenet.init_weights()

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars \
                contains all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, float):
                log_vars[loss_name] = loss_value
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors or float')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if not isinstance(loss_value, float):
                if dist.is_available() and dist.is_initialized():
                    loss_value = loss_value.data.clone()
                    dist.all_reduce(loss_value.div_(dist.get_world_size()))
                log_vars[loss_name] = loss_value.item()
            else:
                log_vars[loss_name] = loss_value

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        input_dict = data['input_dict']
        outputs = dict(
            loss=loss, log_vars=log_vars,
            num_samples=len(input_dict[list(input_dict.keys())[0]]['img_metas']))
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)
        input_dict = data['input_dict']
        outputs = dict(
            loss=loss, log_vars=log_vars,
            num_samples=len(input_dict[list(input_dict.keys())[0]]['img_metas']))
        return outputs
