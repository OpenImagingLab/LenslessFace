from mmcls.models import build_optical
from mmcv import Config
from mmcls.datasets.pipelines import Compose
import os
import cv2 as cv
from torchvision.utils import save_image
from tqdm import tqdm
# cfg = Config.fromfile('configs/distill/face/vit2optical_bg_af_updater_rotate_scale_shift_crop_binary_fix.py')
cfg = Config.fromfile('configs/distill/face/base_12800.py')
cfg.algorithm.architecture.model.backbone.optical.load_weight_path = "logs/distill/face/base_12800/latest.pth"
cfg.algorithm.architecture.model.backbone.optical.expected_light_intensity = 1
cfg.algorithm.architecture.model.backbone.optical.type = "SoftPsfConvDiff"
cfg.algorithm.architecture.model.backbone.optical = dict(
    type='LoadPsf',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    center_crop_size=[240, 200],
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type="gaussian",
    expected_light_intensity=12800,
    # do_affine = True,
    requires_grad_psf = False,
    binary=True,
    # load_psf_path = "/mnt/data/oss_beijing/caixin/psf_square.png",
    load_psf_path = "/root/caixin/RawSense/mmrazor/vis_optical_result/0_psf.png",
    n_psf_mask=1)

cfg.data.val.pipeline = [
            dict(type='LoadImageFromFile'),
            dict(
                type='Propagated',
                keys=['img'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.5,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[172, 172, 3],
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 30),
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=0.0,
                ),
            dict(type='Affine2label',),
            # dict(type='AddBackground', img_dir='/mnt/workspace/RawSense/data/BG-20k/testval',size = (100, 100),is_tensor=True),
            # dict(type='StackImagePair', keys=['img_nopad'], out_key='img'),
            dict(type='Collect', keys=['img', 'affine_matrix','target','target_weight'],meta_keys=['image_file'])
]


def optical_conv(source_img_path, cfg, target_dir = "vis_optical_result", index = None):
    source_filename = os.path.basename(source_img_path)
    source_dir = os.path.dirname(source_img_path)
    pipeline = Compose(cfg.data.val.pipeline)
    
    optical = build_optical(cfg.algorithm.architecture.model.backbone.optical)
    source_dict = dict(img_info=dict(filename=source_filename))
    source_dict['img_prefix'] = source_dir
    source = pipeline(source_dict)
    # print(source["img"].device, optical._psf.device)
    optical(source["img"].to(optical._psf.device))
    # visualize the result 

    after_optical = optical.after_optical
    before_optical = optical.before_optical
    after_affine = optical.after_affine
    # create save path
    os.makedirs(target_dir, exist_ok=True)
    # save the result
    # print("target_dir", target_dir)
    if index is not None:
        save_psf_path = os.path.join(target_dir, '%s_%d_psf.png'%(source_filename.split('.')[0], index))
        after_optical = save_image(after_optical,
                                os.path.join(target_dir, '%s_%d_after_optical.png'%(source_filename.split('.')[0], index)),
                                normalize = True)
    else:
        after_optical = save_image(after_optical,
                                os.path.join(target_dir, '%s.png'%source_filename.split('.')[0]),
                                normalize = True)
        # before_optical = save_image(before_optical,os.path.join(target_dir, '%s_before_optical.png'%source_filename.split('.')[0]), normalize = True)
        # after_affine = save_image(after_affine, os.path.join(target_dir, '%s_after_affine.png'%source_filename.split('.')[0]), normalize = True)
        # conv_weight = optical._psf.cpu().detach().numpy()
        # mask = optical.psf_val_to_save.cpu().detach().numpy()
    #     save_psf_path = os.path.join(target_dir, '%s_psf.png'%source_filename.split('.')[0])
 
    # # save_mask_path = os.path.join(target_dir, '%s_mask.png'%source_filename.split('.')[0])
    # optical.save_psf(save_psf_path)
   
def read_psf(log_path):
    imgs_path = os.path.join(log_path, 'visualizations')
    imgs = os.listdir(imgs_path)
    psf_list = []
    for img in imgs:
        if img.endswith('psf.png'):
            psf_list.append(img)
    #sort psf with order
    psf_list.sort(key=lambda x:int(x.split('_')[0]))
    return psf_list

if __name__ == "__main__":
    
    # source_imgs_path = "/root/caixin/RawSense/mmrazor/vis_optical/point_source"
    # # source_imgs_path = "vis_optical/paper_example"
    # source_imgs_path = "/root/caixin/data/celebrity_single"
    # target_dir = "/root/caixin/data/celebrity_single_optical"
    # source_imgs_path = "/root/caixin/data/lfw/lfw-deepfunneled-single"
    # target_dir = "/root/caixin/data/lfw/lfw-deepfunneled-single-optical"
    source_imgs_path = "/root/caixin/data/lfw/lfw-172X172-single"
    
    target_dir = "/root/caixin/data/lfw/lfw-172x172-single-optical"
    # source_imgs_path = "/root/caixin/data/lfw/lfw-deepfunneled"
    for img in tqdm(os.listdir(source_imgs_path)):
        if img.endswith('.png') or img.endswith('.jpg'):
            source_img_path = os.path.join(source_imgs_path, img)
            optical_conv(source_img_path,cfg, target_dir = target_dir, index = None)

  
