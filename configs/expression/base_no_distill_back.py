import torch
teacher_ckpt = "/root/caixin/RawSense/LenslessFace/logs/expression/rgb_teacher_resnet/epoch_200.pth"
optical = dict(
    type='SoftPsfConv',
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
    expected_light_intensity=12800,
    do_affine = True,
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
    requires_grad=True,
    use_stn=False,
    do_optical=False,
    down="resize",
    noise_type=None,
    do_affine=True,
    n_psf_mask=1)

find_unused_parameters = True
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
img_norm_cfg = dict(
    mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0], to_rgb=True)


data = dict(
    workers_per_gpu=2,
    train=dict(
        type='RafDB',
        img_prefix='data/rafdb/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            # color and brightness jittering
            # dict(type='Contrast', magnitude = 0.2),
            # dict(type='ColorTransform', magnitude = 0.2),
            # dict(type='Brightness', magnitude = 0.2),
            # dict(type='Normalize', **img_norm_cfg),
            dict(
                type='Propagated',
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[256, 256, 3],
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 0),
                    prob=0.0,
                ),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='StackImagePair', keys=['img', 'img_nopad'], out_key='img'),
            dict(type='Collect', keys=['img','gt_label', 'affine_matrix'])
        ]),
    val=dict(
        type='RafDB',
        img_prefix='data/rafdb/',
        # test_mode=False,
        test_mode=True,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', size=(256, 256)),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            # dict(type='Normalize', **img_norm_cfg),
            dict(
                type='Propagated',
                mask2sensor=0.002,
                scene2mask=0.4,
                object_height=0.27,
                sensor='IMX250',
                single_psf=False,
                grayscale=False,
                input_dim=[256, 256, 3],
                output_dim=[308, 257, 3],
                dtype_out=torch.float32),
            dict(
                    type='TorchAffineRTS',
                    angle=(0, 0),
                    prob=0.0,
                ),
            dict(type='ToTensor', keys=['gt_label']),
             dict(type='StackImagePair', keys=['img_nopad'], out_key='img'),
            dict(type='Collect', keys=['img','gt_label', 'affine_matrix'])
        ]),
    train_dataloader=dict(samples_per_gpu=8),
    val_dataloader=dict(samples_per_gpu=64),
    test_dataloader=dict(samples_per_gpu=64))
custom_hooks = [
    dict(type='VisualConvHook'),
    dict(type='VisualAfterOpticalHook')
]

student = dict(
    type='mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='T2T_ViT_optical',
        optical=optical,
        apply_affine=True,
        image_size=256),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=384,
        out_channels=128,
        kernel_size=(16, 16)),
    head=dict(type='LinearClsHead', num_classes = 7, in_channels=128, init_cfg=dict(type='Normal', layer='Linear', std=0.01))
)

teacher = dict(
    type='mmcls.AffineFaceImageClassifier',
    backbone=dict(
        type='ResNet_optical',
        optical=no_optical,
        remove_bg=True,
        image_size=256,
        depth=18,
        num_stages=4,
        out_indices=(3, ),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18'),
        style='pytorch'),
    neck=dict(
        type='GlobalDepthWiseNeck',
        in_channels=512,
        out_channels=128,
        kernel_size=(8, 8)),
    head=dict(
        type='LinearClsHead',
        num_classes=7,
        in_channels=128,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, ),
    ),
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
                        loss_weight=0.0,
                        with_l2_norm=True),
                    dict(
                        type='AngleWiseRKD',
                        name='angle_wise_loss',
                        loss_weight=0.0,
                        with_l2_norm=True),
                ])
        ]),
)


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
runner = dict(type='EpochBasedRunner', max_epochs=200)
evaluation = dict(interval=1, metric='accuracy')
# runner = dict(type='IterBasedRunner', max_iters=200000)
# checkpoint_config = dict(interval=1000)
# evaluation = dict(interval=500,metric='accuracy')
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
