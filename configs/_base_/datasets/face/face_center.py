dataset_info = dict(
    dataset_name='face_center',
    paper_info={},  # Optional, to be displayed in the doc
    keypoint_info={
        0: dict(name='face_center', id=0, color=[255, 255, 255], type='', swap='face_center'),
    },
    skeleton_info={},
    joint_weights=[1.],
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