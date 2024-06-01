import os.path as osp
import pickle
import shutil
import tempfile
from mmcv.image import tensor2imgs
import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
import numpy as np
import cv2
def single_gpu_test(pose_model, cls_model, data_loader, show = False, out_dir=None):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.


    Returns:
        list: The prediction results.
    """

    pose_model.eval()
    cls_model.eval()
    pose_results = []
    cls_results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    idx = 0
    for i, data in enumerate(data_loader):
        # print(data.keys())
        with torch.no_grad():
            pose_result = pose_model(return_loss=False,return_img=True, return_heatmap=True, **data)
        if 'output_heatmap' in pose_result.keys():
            # transfer output_heatmap (shift) to affine matrix for cls_model
            print(data['affine_matrix'][0])
            data['affine_matrix'] = pose_model.module.shift2affine(pose_result['output_heatmap'], data['img'].shape[-2:])
            print(data['affine_matrix'][0])
            # print(data['img'].shape[-2:])
        # print(data['target'][0])
        with torch.no_grad():
            cls_result = cls_model(return_loss=False, **data)
        if isinstance(cls_result, list):
            cls_results.extend(cls_result)
        else:
            cls_results.append(cls_result)
        # use the first key as main key to calculate the batch size
        batch_size = len(data['img'])
        target = data['target']
        pose_result['target'] = target
        for _ in range(batch_size):
            prog_bar.update()
            # imgs = tensor2imgs(img_tensor)
            #trans img_tensor into imgs (ndarray)
        if show or out_dir:
            img_tensor = pose_result['img']
            img_metas = pose_result['img_metas']
            imgs = []
            img = img_tensor.cpu().detach().numpy()
            img = np.transpose(img, (0, 2, 3, 1))
            for i in range(img.shape[0]):
                # Normalize image to [0, 1] range
                img_normalized = (img[i] - img[i].min()) / (img[i].max() - img[i].min())
                # Scale normalized image to [0, 255] range and convert data type to integer
                img_scaled = (img_normalized * 255).astype(np.uint8)
                # transfer img_scaled to BGR
                img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_RGB2BGR)
                imgs.append(img_scaled)
            assert len(imgs) == len(img_metas)
            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                # h, w, _ = img_meta['img_shape']
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['image_file'].split("data/")[-1].split(".")[0] + '_%d.jpg'%idx)
                    idx += 1
                    # print(out_file)
                else:
                    out_file = None
                # print(result.keys())
                pose_model.module.show_internal_result(
                    img,
                    pose_result['output_heatmap'][i],
                    target = target[i],
                    out_file=out_file)
            pose_result.pop('img')
        pose_results.append(pose_result)
    return pose_results, cls_results    

def multi_gpu_test():
    #TODO
    pass