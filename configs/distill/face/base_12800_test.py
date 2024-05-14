
teacher_ckpt = "checkpoints/rgb_teacher/epoch_40.pth"
optical = dict(
    type='SoftPsfConvDiff',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[240, 200],
    center_crop_size=[240, 200],
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type="gaussian",
    expected_light_intensity=12800 ,
    do_affine = True,
    load_weight_path="/root/caixin/RawSense/mmrazor/logs/distill/face/base_12800/latest.pth",
    # requires_grad_psf = False,
    binary=True,
    n_psf_mask=1)
no_optical = dict(
    type='SoftPsfConv',
    feature_size=2.76e-05,
    sensor='IMX250',
    input_shape=[3, 308, 257],
    scene2mask=0.4,
    mask2sensor=0.002,
    target_dim=[240, 200],
    do_optical=False,
    requires_grad=True,
    use_stn=False,
    down="resize",
    noise_type=None,
    n_psf_mask=1)


student = dict(
    type = 'mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=optical,
        image_size=240),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(15, 15)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)))
teacher = dict(
    type = 'mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=no_optical,
        # apply_affine=True,
        image_size=240,
        remove_bg=True),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(15, 15)),
    head=dict(
        type='IdentityClsHead',
        loss=dict(type='ArcMargin', out_features=93955)),
    init_cfg=dict(type='Pretrained', checkpoint=teacher_ckpt, map_location='cpu'),
)

algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMClsArchitecture',
        model=student,
    ),
    with_student_loss=True,
    with_teacher_loss=False,
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        teacher_norm_eval=True,
        components=[
            dict(
                student_module='neck.fc',
                teacher_module='neck.fc',
                  losses=[
                    dict(
                        type='DistanceWiseRKD',
                        name='distance_wise_loss',
                        loss_weight=100.0,
                        with_l2_norm=True),
                    dict(
                        type='AngleWiseRKD',
                        name='angle_wise_loss',
                        loss_weight=200.0,
                        with_l2_norm=True),
                ])
        ]),
)
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook'),
    dict(type='BGUpdaterHook', max_progress=0.2),
    dict(type='AffineUpdaterHook',max_progress=0.2,
    apply_translate=True,
    apply_scale=True),
]


find_unused_parameters = True
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
dataset_type = 'Celeb'
num_classes = 93955

train_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(172, 172)),
            dict(type='Pad_celeb', size=(180, 172), padding=(0, 8, 0, 0)),
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
                output_dim=[308, 257, 3]),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 0),
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=1.0,
                ),
            dict(type='AddBackground', img_dir='data/BG-20k/train',size = (100, 100)),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='StackImagePair', keys=['img', 'img_nopad'], out_key='img'),
            dict(type='Collect', keys=['img', 'gt_label', 'affine_matrix'])
        ]
val_pipeline = [
            dict(type='LoadImageFromFile'),
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
                    angle=(0, 30),
                    scale_factor=0.2,
                    translate=(0.2, 0.2),
                    prob=1.0,
                ),
            dict(type='Affine2label',),
            # dict(type='AddBackground', img_dir='data/BG-20k/testval',size = (100, 100),is_tensor=True),
            dict(type='Collect', keys=['img', 'affine_matrix','target','target_weight'],meta_keys=['image_file'])
]
test_pipeline = [
            dict(type='LoadImagePair'),
            dict(
                type='FlipPair',
                keys=['img1', 'img2'],
                keys_flip=['img1_flip', 'img2_flip']),
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
                output_dim=[308, 257, 3]),
   
            dict(type="TorchAffineRTS",
                translate = (0.2,0.2),
                angle=(0, 0),
                return_translate=True,
                scale_factor=0.2,
                prob=1.0),
   
            dict(type='ToTensor', keys=['fold', 'label']),
            dict(
                type='StackImagePair',
                keys=['img1', 'img1_flip', 'img2', 'img2_flip'],
                out_key='img'),
            dict(type='Collect', keys=['img', 'fold', 'label', 'affine_matrix'])
        ]
test_pipeline = val_pipeline

data = dict(
    workers_per_gpu=2,
    train=dict(
        type='Celeb',
        img_prefix='data/celebrity/',
        imglist_root=
        'data/celebrity/celebrity_data.txt',
        label_root='data/celebrity/celebrity_label.txt',
        pipeline=train_pipeline),
    val=dict(
        type='LFW',
            load_pair = False,
            use_flip = False,
            img_prefix='data/lfw/lfw-112X96',
            pair_file='data/lfw/pairs.txt',
        pipeline=val_pipeline),
    test=dict(
        type='LFW',
            load_pair = False,
            use_flip = False,
            img_prefix='data/lfw/lfw-112X96',
            pair_file='data/lfw/pairs.txt',
            pipeline=test_pipeline
   ),
    train_dataloader=dict(samples_per_gpu=70, persistent_workers=False),
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32))

optimizer = dict(type='AdamW',lr=5e-4, weight_decay=0.05)
lr_config = dict(
    policy='CosineAnnealingCooldown',
    min_lr=1e-5,
    cool_down_time=10,
    cool_down_ratio=0.1,
    by_epoch=True,
    warmup_by_epoch=True,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=1e-6)
checkpoint_config = dict(interval=10)
runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='accuracy')
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))