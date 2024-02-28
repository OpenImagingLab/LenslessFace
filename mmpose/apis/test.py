# Copyright (c) OpenMMLab. All rights reserved.
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
def single_gpu_test(model, data_loader,
                   return_img=True,
                   out_dir=None):
    """Test model with a single gpu.

    This method tests model with a single gpu and displays test progress bar.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.


    Returns:
        list: The prediction results.
    """

    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    idx = 0
    for i, data in enumerate(data_loader):
        # print(data.keys())
        with torch.no_grad():
            result = model(return_loss=False,return_img=return_img, return_heatmap=True, **data)
            
        # use the first key as main key to calculate the batch size
        batch_size = len(data['img'])
        for _ in range(batch_size):
            prog_bar.update()
        if out_dir:
            img_tensor = result['img']
            img_metas = result['img_metas']
            target = data['target']
            result['target'] = target
            # imgs = tensor2imgs(img_tensor)
            #trans img_tensor into imgs (ndarray)
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
                    # print(out_file)
                else:
                    out_file = None
                # print(result.keys())
                model.module.show_internal_result(
                    img,
                    result['output_heatmap'][i],
                    target = target[i],
                    out_file=out_file)
        result.pop('img')
        results.append(result)
    return results
def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for data in data_loader:
        with torch.no_grad():
            result = model(return_loss=False, return_heatmap=True, **data)
        results.append(result)

        if rank == 0:
            # use the first key as main key to calculate the batch size
            # print(len(next(iter(data.values()))))
            # batch_size = len(next(iter(data.values())))
            batch_size = len(data['img'])
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    """Collect results in cpu mode.

    It saves the results on different gpus to 'tmpdir' and collects
    them by the rank 0 worker.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode. Default: None

    Returns:
        list: Ordered results.
    """
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # synchronizes all processes to make sure tmpdir exist
    dist.barrier()
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    # synchronizes all processes for loading pickle file
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None

    # load results of all parts from tmp dir
    part_list = []
    for i in range(world_size):
        part_file = osp.join(tmpdir, f'part_{i}.pkl')
        part_list.append(mmcv.load(part_file))
    # sort the results
    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    # the dataloader may pad some samples
    ordered_results = ordered_results[:size]
    # remove tmp dir
    shutil.rmtree(tmpdir)
    return ordered_results


def collect_results_gpu(result_part, size):
    """Collect results in gpu mode.

    It encodes results to gpu tensors and use gpu communication for results
    collection.

    Args:
        result_part (list): Results to be collected
        size (int): Result size.

    Returns:
        list: Ordered results.
    """

    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
    return None
