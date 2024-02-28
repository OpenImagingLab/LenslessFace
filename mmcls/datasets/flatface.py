import numpy as np
import scipy.misc
import torch
from abc import ABCMeta, abstractmethod
from torch.utils.data import Dataset
import os
from .base_dataset import BaseDataset
from .pipelines import Compose
from .builder import DATASETS


def getAccuracy(scores, flags, threshold):
    p = (scores[flags == 1] > threshold).sum()
    n = (scores[flags == -1] < threshold).sum()
    return 1.0 * (p + n) / len(scores)


def getThreshold(scores, flags, thrNum):
    accuracys = torch.zeros((2 * thrNum + 1, 1))
    thresholds = torch.arange(-thrNum, thrNum + 1) * 1.0 / thrNum
    for i in range(2 * thrNum + 1):
        accuracys[i] = getAccuracy(scores, flags, thresholds[i])
    max_index = (accuracys == accuracys.max()).squeeze()
    bestThreshold = thresholds[max_index].mean()
    return bestThreshold


@DATASETS.register_module()
class FlatFace(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 img_prefix,
                 pair_file,
                 pipeline,
                 load_pair=True,
                 test_mode=False,
                 use_flip=True):

        super(FlatFace, self).__init__()
        self.pipeline = Compose(pipeline)
        self.test_mode = test_mode
        self.img_prefix = img_prefix
        self.pair_file = pair_file
        self.use_flip = use_flip
        self.load_pair = load_pair
        data_infos = []
        if self.load_pair:
            with open(pair_file) as f:
                pairs = f.read().splitlines()[:]
            for i, line in enumerate(pairs):
                p = line.split('\t')[0].split(' ')
                imgfile1 = os.path.join(self.img_prefix, p[0])
                imgfile2 = os.path.join(self.img_prefix, p[1])
                person1 = p[0].split('/')[0]
                person2 = p[1].split('/')[0]
                if person1 == person2:
                    label = 1
                else:
                    label = -1

                fold = i // 2000
                #print(fold)
                info = {'imgfile1': imgfile1, 'imgfile2': imgfile2, 'label': label, 'fold': fold}
                #print(info)
                data_infos.append(info)
        # if not load pair, then load all images one by one
        if not load_pair:
            for person in os.listdir(self.img_prefix):
                for i, imgfile in enumerate(os.listdir(os.path.join(self.img_prefix, person))):
                    imgfile = os.path.join(self.img_prefix, person, imgfile)
                    info = dict(img_info=dict(filename=os.path.join(person, imgfile)))
                    info['img_prefix'] = self.img_prefix
                    data_infos.append(info)

        self.data_infos = data_infos

        # # for debug
        # from scipy.io import loadmat
        # data = loadmat('/home/SENSETIME/zengwang/mydata/best_result.mat')
        # results = []
        # fold = data['fold']
        # flag = data['flag']
        # fr = data['fr']
        # fl = data['fl']
        # for i in range(flag.shape[1]):
        #     res = {}
        #     feature = [torch.tensor(fl[i]), torch.tensor(fr[i])]
        #     feature = torch.stack(feature, dim=0)
        #     feature = feature.reshape(1, 4, 128)
        #     res['feature'] = feature
        #     res['fold'] = torch.tensor(fold[0, i]).unsqueeze(-1).unsqueeze(-1)
        #     res['label'] = torch.tensor(flag[0, i]).unsqueeze(-1).unsqueeze(-1)
        #     results.append(res)
        # self.evaluate(results)

    def __getitem__(self, index):
        data_info = self.data_infos[index]
        return self.pipeline(data_info)

    def __len__(self):
        return len(self.data_infos)

    def evaluate_with_pair(self,
                 results,
                 logger=None):
     
        accs = torch.zeros(10)
        feature = [res['feature'] for res in results]
        fold = [res['fold'] for res in results]
        label = [res['label'] for res in results]

        feature = torch.cat(feature, dim=0)
        if self.use_flip:
            feature1 = torch.cat([feature[:, 0], feature[:, 1]], dim=-1)
            feature2 = torch.cat([feature[:, 2], feature[:, 3]], dim=-1)
        else:
            feature1 = feature[:, 0]
            feature2 = feature[:, 1]
        folds = torch.cat(fold, dim=0).squeeze()
        labels = torch.cat(label, dim=0).squeeze()
        for i in range(10):
            valfold = folds != i
            testfold = folds == i
            mean_val = torch.cat([
                feature1[valfold], feature2[valfold]],
                dim=0).mean(dim=0, keepdim=True)

            fea1 = feature1 - mean_val
            fea2 = feature2 - mean_val
            fea1 = fea1 / fea1.norm(p=2, dim=-1, keepdim=True)
            fea2 = fea2 / fea2.norm(p=2, dim=-1, keepdim=True)
            scores = (fea1 * fea2).sum(dim=-1)
            threshold = getThreshold(scores[valfold], labels[valfold], 10000)
            accs[i] = getAccuracy(scores[testfold], labels[testfold], threshold)
        for i in range(len(accs)):
            logger.info('{}    {:.2f}'.format(i + 1, accs[i] * 100))
        logger.info('--------')
        logger.info('AVE    {:.2f}'.format(accs.mean() * 100))
        eval_result = dict(accuracy=accs.numpy())
        return eval_result
    
    def evaluate_with_single(self, results, 
                 logger=None):
        imgs_list = [res['img_metas']['image_file'] for res in results]
        feature = [res['feature'] for res in results]
        labels = []
        feature_for_pair = []
        folds = []
        with open(self.pair_file) as f:
            pairs = f.read().splitlines()[1:]
        for i, line in enumerate(pairs):
            p = line.split('\t')[0].split(' ')

            imgfile1 = os.path.join(self.img_prefix, p[0])
            imgfile2 = os.path.join(self.img_prefix, p[1])
            person1 = p[0].split('/')[0]
            person2 = p[1].split('/')[0]
            if person1 == person2:
                labels.append(torch.tensor([[1]]))
            else:
                labels.append(torch.tensor([[-1]]))
                
     
            fold = i // 2000

            folds.append(torch.tensor([[fold]]))
            #concat feature1 and feature2
            feature1 = feature[imgs_list.index(imgfile1)]
            feature2 = feature[imgs_list.index(imgfile2)]

            feature_for_pair.append(torch.stack([feature1, feature2], dim=1))
        result_for_evaluation = []
        for i in range(len(pairs)):
            result_for_evaluation.append({"feature": feature_for_pair[i], "fold": folds[i], "label": labels[i]})
        return self.evaluate_with_pair(result_for_evaluation, logger)


    def evaluate(self, results, res_folder=None, metric='accuracy',metric_options=None,
                 logger=None):
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy', 'NME']
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        if res_folder is not None:
            self.save_results(results, res_folder)
        if 'accuracy' in metrics:
            if self.load_pair:
                return self.evaluate_with_pair(results, logger)
            else:
                return self.evaluate_with_single(results, logger)
     
        if 'NME' in metrics:
            return self.evaluate_pose(results, metric='NME')
    #save results for evaluation
    def save_results(self, results, res_file):
        imgs_list = [res['img_metas']['image_file'] for res in results]
        feature = [res['feature'] for res in results]
        labels = []
        # form imgs_list parsing labels
        for img in imgs_list:
            # print(img)
            labels.append(int(img.split('/')[-2]))
        if res_file.endswith(".pkl"):
            import pickle
            with open(res_file, 'wb') as f:
                pickle.dump({"feature": feature, "label": labels}, f)
                print("save results to {}".format(res_file))



    def evaluate_both(self, pose_results, cls_results, res_folder=None, metric=['accuracy', 'NME'], metric_options=None,
                 logger=None):
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = ['accuracy', 'NME']
        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')
        if 'accuracy' in metrics:
            if self.load_pair:
                self.evaluate_with_pair(cls_results, logger)
            else:
                self.evaluate_with_single(cls_results, logger)
        if 'NME' in metrics:
            return self.evaluate_pose(pose_results, metric='NME')
    
    def _eval_nme(self, preds, targets):
        # calculate normalized mean error
        distances = np.linalg.norm(preds - targets, axis=-1)
        nme = np.mean(distances)
        return nme
    
    def evaluate_pose(self, results, res_folder=None, metric='NME', **kwargs):
        """Evaluate freihand keypoint results. The pose prediction results will
        be saved in ``${res_folder}/result_keypoints.json``.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmap height: H
            - heatmap width: W

        Args:
            results (list[dict]): Testing results containing the following
                items:

                - preds (np.ndarray[1,K,3]): The first two dimensions are \
                    coordinates, score is the third dimension of the array.
                - boxes (np.ndarray[1,6]): [center[0], center[1], scale[0], \
                    scale[1],area, score]
                - image_path (list[str]): For example, ['wflw/images/\
                    0--Parade/0_Parade_marchingband_1_1015.jpg']
                - output_heatmap (np.ndarray[N, K, H, W]): model outputs.
            res_folder (str, optional): The folder to save the testing
                results. If not specified, a temp folder will be created.
                Default: None.
            metric (str | list[str]): Metric to be performed.
                Options: 'NME'.

        Returns:
            dict: Evaluation results for evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['NME']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        preds = []
        targets = []
        for result in results:
            # print(result['output_heatmap'])
            result_targets = result['target']
            output_heatmaps = result['output_heatmap']
            #if result_targets is tensor, transform to numpy
            if isinstance(result_targets, torch.Tensor):
                result_targets = result_targets.cpu().numpy()
                
            for i, (output_heatmap, target) in enumerate(zip(output_heatmaps, result_targets)):
                # print(img_meta["affine_matrix"])
                # target = img_meta["affine_matrix"][:,2]
                # target = np.array([0.5 + translate[0],0.5 + translate[1]]).reshape(1,2)
                preds.append(output_heatmap)
                targets.append(target)
            # print(target, output_heatmap)
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        res = {}
        if 'NME' in metrics:
            res['NME'] = self._eval_nme(preds, targets)
        print(res)
        return res