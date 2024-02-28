# Copyright (c) OpenMMLab. All rights reserved.
from mmrazor.models.builder import ALGORITHMS
from mmrazor.models.utils import add_prefix
from .base import BaseAlgorithm
import copy
@ALGORITHMS.register_module()
class GeneralDistill(BaseAlgorithm):
    """General Distillation Algorithm.

    Args:
        with_student_loss (bool): Whether to use student loss.
            Defaults to True.
        with_teacher_loss (bool): Whether to use teacher loss.
            Defaults to False.
    """

    def __init__(self,
                 with_student_loss=True,
                 with_teacher_loss=False,
                 **kwargs):

        super(GeneralDistill, self).__init__(**kwargs)
        self.with_student_loss = with_student_loss
        self.with_teacher_loss = with_teacher_loss

    def train_step(self, data, optimizer):
        """"""
        # student_data = copy.deepcopy(data)
        losses = dict()
        if self.with_teacher_loss:
            teacher_losses = self.distiller.exec_teacher_forward(data)
            teacher_losses = add_prefix(teacher_losses, 'teacher')
            losses.update(teacher_losses)
          
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_teacher_forward(data)
        if self.with_student_loss:
            student_losses = self.distiller.exec_student_forward(
                self.architecture, data)
            student_losses = add_prefix(student_losses, 'student')
            losses.update(student_losses)
        else:
            # Just to be able to trigger the forward hooks that
            # have been registered
            _ = self.distiller.exec_student_forward(self.architecture, data)
       

        distill_losses = self.distiller.compute_distill_loss(data)
        distill_losses = add_prefix(distill_losses, 'distiller')
        losses.update(distill_losses)

        loss, log_vars = self._parse_losses(losses)
        if 'img_metas' not in data.keys():
            input_dict = data['input_dict']
            num_samples = len(input_dict[list(input_dict.keys())[0]]['img_metas'])
          
        else:
            num_samples = len(data['img_metas'])
        outputs = dict(
                loss=loss, log_vars=log_vars,
                num_samples=num_samples)
        return outputs

    def val_step(self, data, optimizer=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        if 'img_metas' not in data.keys():
            input_dict = data['input_dict']
            num_samples = len(input_dict[list(input_dict.keys())[0]]['img_metas'])
        else:
            num_samples = len(data['img_metas'])
        outputs = dict(
                loss=loss, log_vars=log_vars,
                num_samples=num_samples)
        
        return outputs