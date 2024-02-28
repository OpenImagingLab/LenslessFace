# Copyright (c) OpenMMLab. All rights reserved.
import time
import warnings
import mmcv
from mmcv.runner import RUNNERS, IterBasedRunner, get_host_info, IterLoader
import random
class HybridIterLoader:

    def __init__(self, dataloader_dict, main_key=None):
        self._iter_loaders = {}
        for key, dataloader in dataloader_dict.items():
            if dataloader is not None:
                self._iter_loaders[key] = IterLoader(dataloader)
        self.main_key = list(self._iter_loaders.keys())[0] if main_key is None else main_key

    @property
    def epoch(self):
        return self._iter_loaders[self.main_key].epoch

    def __next__(self):
        data = {}
        for key, iter_loader in self._iter_loaders.items():
            data[key] = iter_loader.__next__()
        return dict(input_dict=data)

    def __len__(self):
        return self._iter_loaders[self.main_key].__len__()


@RUNNERS.register_module()
class HybridIterBasedRunner(IterBasedRunner):
    """Iteration-based Runner.

    This runner train models iteration by iteration.
    """

        # print(conv_weight.shape)

    def run(self, data_loaders, workflow, max_iters=None, **kwargs):
        """Start running.

        Args:
            data_loaders (list[:obj:`DataLoader`]): Dataloaders for training
                and validation.
            workflow (list[tuple]): A list of (phase, iters) to specify the
                running order and iterations. E.g, [('train', 10000),
                ('val', 1000)] means running 10000 iterations for training and
                1000 iterations for validation, iteratively.
        """
        assert isinstance(data_loaders, list)
        assert mmcv.is_list_of(workflow, tuple)
        assert len(data_loaders) == len(workflow)
        if max_iters is not None:
            warnings.warn(
                'setting max_iters in run is deprecated, '
                'please set max_iters in runner_config', DeprecationWarning)
            self._max_iters = max_iters
        assert self._max_iters is not None, (
            'max_iters must be specified during instantiation')

        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s',
                         get_host_info(), work_dir)
        self.logger.info('Hooks will be executed in the following order:\n%s',
                         self.get_hook_info())
        self.logger.info('workflow: %s, max: %d iters', workflow,
                         self._max_iters)
        self.call_hook('before_run')
        # random shuffle the dataloader
        iter_loaders = [HybridIterLoader(x) if isinstance(x, dict) else IterLoader(x) for x in data_loaders]

        self.call_hook('before_epoch')

        while self.iter < self._max_iters:
            for i, flow in enumerate(workflow):
                self._inner_iter = 0
                mode, iters = flow
                if not isinstance(mode, str) or not hasattr(self, mode):
                    raise ValueError(
                        'runner has no method named "{}" to run a workflow'.
                        format(mode))
                iter_runner = getattr(self, mode)
                for _ in range(iters):
                    if mode == 'train' and self.iter >= self._max_iters:
                        break
                    iter_runner(iter_loaders[i], **kwargs)
                    
        time.sleep(1)  # wait for some hooks like loggers to finish
        self.call_hook('after_epoch')
        self.call_hook('after_run')





