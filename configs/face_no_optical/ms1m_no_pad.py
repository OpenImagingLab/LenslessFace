import torch
optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[164, 128],
    requires_grad=True,
    use_stn=False,
    do_optical=False,
    down="resize",
    noise_type=None,
    do_affine=True,
    n_psf_mask=1)
propagated_args = dict(
    mask2sensor=0.002,
    scene2mask=0.4,
    object_height=0.27,
    sensor='IMX250',
    single_psf=False,
    grayscale=False,
    input_dim=[112, 96, 3],
    output_dim=[308, 257, 3],
    dtype_out=torch.float32)
find_unused_parameters = True
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'Celeb'
num_classes = 93431
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)

data = dict(
    workers_per_gpu=8,
    # timeout=30,
    train=dict(
        type='MXFaceDataset',
        data_root='/mnt/caixin/RawSense/data/ms1m-retinaface-t1',
        pipeline=[
            # dict(type='LoadImageFromFile'),
            dict(type='CenterCrop', crop_size=(112, 96)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Propagated',
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
         
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 0),
                    # scale_factor=0.2,
                    # translate=(0.2, 0.2),
                    prob=1.0,
                ),
            dict(type='ToTensor', keys=['gt_label']),
             dict(type='StackImagePair', keys=['img_nopad'], out_key='img'),
            dict(type='Collect', keys=['img','gt_label', 'affine_matrix'])
        ]),
    val=dict(
        type='LFW',
        img_prefix='/mnt/caixin/RawSense/data/lfw/lfw-112X96',
        pair_file='/mnt/caixin/RawSense/data/lfw/pairs.txt',
        pipeline=[
            dict(type='LoadImagePair'),
            dict(
                type='FlipPair',
                keys=['img1', 'img2'],
                keys_flip=['img1_flip', 'img2_flip']),
            # dict(type='AffineRTS', angle=45.0, prob=1.0),
            # dict(type='ImageToTensor', keys=['img1', 'img1_flip', 'img2', 'img2_flip']),
           
            dict(
                type='Propagated',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 0),
                    # scale_factor=0.2,
                    # translate=(0.2, 0.2),
                    prob=1.0,
                ),
          
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1_nopad', 'img1_flip_nopad', 'img2_nopad', 'img2_flip_nopad'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label', 'affine_matrix'])
        ]),
    test=dict(
        type='IJB_C',
        data_root='/mnt/caixin/RawSense/data/ijb',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='CenterCrop', crop_size=(112, 96)),
            # dict(type='AffineRTS', angle=45.0, prob=1.0),
            # dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Propagated',
                keys=['img'],
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[112, 96, 3],
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 0),
                    # scale_factor=0.2,
                    # translate=(0.2, 0.2),
                    prob=1.0,
                ),
            dict(type='Affine2label',),
            dict(
                type='StackImagePair',
                keys=['img_nopad'],
                out_key='img'),
            # dict(type='Collect', keys=['img'],)
            dict(type='Collect', clear_res = True, keys=['img', 'affine_matrix','target','target_weight'],)
            
            # meta_keys=['image_file','affine_matrix'])
        ]),
    train_dataloader=dict(samples_per_gpu=200),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=400))
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook')
]
model = dict(
    type='AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=optical,
        apply_affine=True,
        image_size=168),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(11, 11)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93431)))
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=20000,
    warmup_ratio=0.25)
checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='accuracy')
# runner = dict(type='IterBasedRunner', max_iters=200000)
# checkpoint_config = dict(interval=1000)
# evaluation = dict(interval=500,metric='accuracy')
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
