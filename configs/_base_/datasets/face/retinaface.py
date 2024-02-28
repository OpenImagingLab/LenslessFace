dataset_info = dict(
    dataset_name='retinaface',
    paper_info=dict(
        author='Deng, Jiankang and Guo, Jia and Ververas, Evangelos and '
        'Kotsia, Irene and Zafeiriou, Stefanos',
        title='RetinaFace: Single-stage Dense Face Localisation in the Wild',
        container='arXiv:1905.00641',
        year='2019',
    ),
    keypoint_info={
        0: dict(name='left_eye', id=0, color=[255, 255, 255], type='', swap='right_eye'),
        1: dict(name='right_eye', id=1, color=[255, 0, 255], type='', swap='left_eye'),
        2: dict(name='nose', id=2, color=[0, 255, 255], type='', swap=''),
        3: dict(name='mouth_left', id=3, color=[0, 0, 255], type='', swap='mouth_right'),
        4: dict(name='mouth_right', id=4, color=[255, 0, 0], type='', swap='mouth_left'),
    },
    skeleton_info={},
    joint_weights=[1.] * 5,
    sigmas=[])  

checkpoint_config = dict(interval=10)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])

log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
