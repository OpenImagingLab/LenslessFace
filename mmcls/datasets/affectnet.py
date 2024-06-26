import torch
from torch.utils.data import Dataset
import os
from .pipelines import Compose
from .builder import DATASETS
import numpy as np
from mmcls.models.losses import accuracy

@DATASETS.register_module()
class AffectNet(Dataset):
    CLASSES = 7

    def __init__(self, img_prefix, pipeline, test_mode=False):
        # print("rafdb init")
        # print("test_mode", test_mode)
        super(AffectNet, self).__init__()
        self.pipeline = Compose(pipeline)
        self.img_prefix = img_prefix
        self.data = []
        # if test_mode:
        #     data_path = os.path.join(img_prefix, "val")
        #     data_mode = "val"
        # else:
        #     data_path = os.path.join(img_prefix, "train")
        #     data_mode = "train"
        data_path = img_prefix
        labels = os.listdir(data_path)
        for label in labels:
            img_list = os.listdir(os.path.join(data_path, label))
            for img in img_list:
                self.data.append((os.path.join( label, img),int(label)))
            

    def __getitem__(self, index):
        img_info = dict(
            filename=self.data[index][0]
            )
        gt_label = self.data[index][1]
        # print("gt_label", gt_label, np.eye(self.CLASSES)[gt_label])
        # print("filename", self.data[index][0])

        #transfer to one-hot
        gt_label = np.eye(self.CLASSES)[gt_label]

        data = dict(
            img_prefix=self.img_prefix, 
            img_info=img_info,
            gt_label=gt_label)
        data = self.pipeline(data)
        return data

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array([data[1] for data in self.data])
        return gt_labels

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1,)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        result_pred = []
        imgs_list = [res['img_metas']['ori_filename'] for res in results]
        #print 10 samples of imgs_list
   

        # labels = [res['gt_label'] for res in results]
        for result in results:
            result_pred.append(result['pred'].cpu().numpy())
          
        results = np.vstack(result_pred)
        gt_labels = self.get_gt_labels()

        #randomly select 10 samples to print
        indice_ = np.random.choice(len(imgs_list), 10, replace=False)
        # for i in range(10):
        #     print("imgs_list", imgs_list[indice_[i]])
        #     print("gt_labels", gt_labels[indice_[i]])
        #     print("results", results[indice_[i]])


        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
        
    

    def __len__(self):
        return len(self.data)