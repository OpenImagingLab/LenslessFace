# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmcls.datasets.pipelines import Compose
from mmcls.models import build_classifier
from skimage import transform as trans
import cv2

def init_model(config, checkpoint=None, device='cuda:0', options=None):
    """Initialize a classifier from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        options (dict): Options to override some settings in the used config.

    Returns:
        nn.Module: The constructed classifier.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if options is not None:
        config.merge_from_dict(options)
    config.model.pretrained = None
    model = build_classifier(config.model)
    if checkpoint is not None:
        # Mapping the weights to GPU may cause unexpected video memory leak
        # which refers to https://github.com/open-mmlab/mmdetection/pull/6405
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            from mmcls.datasets import ImageNet
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use imagenet by default.')
            model.CLASSES = ImageNet.CLASSES
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


def inference_hybrid_cls_model(model, config, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = mmcv.Config.fromfile(config)
    device = next(model.parameters()).device  # model device
   # cfg.data.test.pipeline=cfg.data.train.pipeline
    #test_pipeline = Compose(cfg.data.train.pipeline)
    #print(test_pipeline)
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
            print('11')
        print('22')
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    print(test_pipeline)
    data = test_pipeline(data)
   # data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    model.eval()
    # forward the model
    with torch.no_grad():
        #scores = model(return_loss=False, **data)
        #pred_score = np.max(scores, axis=1)[0]
        #pred_label = np.argmax(scores, axis=1)[0]
        feature = model(return_loss=False, **data)
        result = {'feature': feature}
    #result['pred_class'] = model.CLASSES[result['pred_label']]
    return result

def inference_hybrid_cls_model_1(model, config, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = mmcv.Config.fromfile(config)
    device = next(model.parameters()).device  # model device
    cfg.data.test.pipeline=cfg.data.train.pipeline
    #test_pipeline = Compose(cfg.data.train.pipeline)
    #print(test_pipeline)
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
            print('11')
        print('22')
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    #print(test_pipeline)
    data = test_pipeline(data)
   # print(data)    
#data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    data['img'] = data['img'].unsqueeze(0)
    print(data['img'].shape)
    model.eval()
    # forward the model
    with torch.no_grad():
        #scores = model(return_loss=False, **data)
        #pred_score = np.max(scores, axis=1)[0]
        #pred_label = np.argmax(scores, axis=1)[0]
        feature = model(return_loss=False, **data)
        result = {'feature': feature}
    #result['pred_class'] = model.CLASSES[result['pred_label']]
    return result





def inference_hybrid_cls_model_2(model,model2, config, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = mmcv.Config.fromfile(config)
    device = next(model.parameters()).device  # model device
    cfg.data.test.pipeline=cfg.data.train.pipeline
    #test_pipeline = Compose(cfg.data.train.pipeline)
    #print(test_pipeline)
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
    #        print('11')
   #     print('22')
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    print(test_pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
   # print(data)
    for img in data['img']:
    #    print(img)
        print(img.shape)
        img = img.unsqueeze(0) 
        img = model2(img)
        data['img'] = img
   # print(data)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        #scores = model(return_loss=False, **data)
        #pred_score = np.max(scores, axis=1)[0]
        #pred_label = np.argmax(scores, axis=1)[0]
        feature = model(return_loss=False, **data)
        result = {'feature': feature}
    #result['pred_class'] = model.CLASSES[result['pred_label']]
    return result




def inference_model(model, img):
    """Inference image(s) with the classifier.
    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.
    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    print(test_pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        pred_score = np.max(scores, axis=1)[0]
        pred_label = np.argmax(scores, axis=1)[0]
        result = {'pred_label': pred_label, 'pred_score': float(pred_score)}
    result['pred_class'] = model.CLASSES[result['pred_label']]
    return result

def show_result_pyplot(model,
                       img,
                       result,
                       fig_size=(15, 10),
                       title='result',
                       wait_time=0):
    """Visualize the classification results on the image.

    Args:
        model (nn.Module): The loaded classifier.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The classification result.
        fig_size (tuple): Figure size of the pyplot figure.
            Defaults to (15, 10).
        title (str): Title of the pyplot figure.
            Defaults to 'result'.
        wait_time (int): How many seconds to display the image.
            Defaults to 0.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        show=True,
        fig_size=fig_size,
        win_name=title,
        wait_time=wait_time)

def read_image(img_path, **kwargs):
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    if mode == 'gray':
        img = cv2.imread(img_path, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    else:
        img = cv2.imread(img_path)
        #if mode == 'rgb':
            # print('to rgb')
        #    img = img[..., ::-1]
        #if layout == 'CHW':
        #    img = np.transpose(img, (2, 0, 1))
    return img

def preprocess(img, bbox=None, landmark=None, **kwargs):

    " warpaffine the test image using the keypoints and bbox of the image"
    if isinstance(img, str):
        img = read_image(img, **kwargs)
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)  #img_size=[112 96]
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        box = [0]*4
        box[0] = bbox[0]['bbox'][0]
        box[1] = bbox[0]['bbox'][1]
        box[2] = bbox[0]['bbox'][2]
        box[3] = bbox[0]['bbox'][3]
        dst[:,0] -= box[0]
        dst[:,1] -= box[1]
        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)
        print(M)
    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        #bb[0] = np.maximum(det[0] - margin / 2, 0)
        #bb[1] = np.maximum(det[1] - margin / 2, 0)
        #bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        #bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        bb[0] = det[0]
        bb[1] = det[1]
        bb[2] = det[2]
        bb[3] = det[3]
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2

        # src = src[0:3,:]
        # dst = dst[0:3,:]

        # print(src.shape, dst.shape)
        # print(src)
        # print(dst)
        # print(M)
        warped = cv2.warpAffine(img[int(box[1]):int(box[3]),int(box[0]):int(box[2])], M, (image_size[1], image_size[0]), borderValue=0.0)

        # tform3 = trans.ProjectiveTransform()
        # tform3.estimate(src, dst)
        # warped = trans.warp(img, tform3, output_shape=_shape)
        return  warped
