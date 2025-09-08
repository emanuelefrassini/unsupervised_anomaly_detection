import os

# Define input and output directories
pkl_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Stick Figures\Stick Figures RDG\WallS\Final procedures\Pyskl_data\dataset_10seconds"
config_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Stick Figures\Stick Figures RDG\WallS\Final procedures\Pyskl_data\configuration_rgbpose"

# Ensure the output directory exists
os.makedirs(config_dir, exist_ok=True)

# Base configuration template
config_template = """# -*- coding: utf-8 -*-
# model_cfg
backbone_cfg = dict(
    type='RGBPoseConv3D',
    speed_ratio=4,
    channel_ratio=4,
    rgb_pathway=dict(
        num_stages=4,
        lateral=True,
        lateral_infl=1,
        lateral_activate=[0, 0, 1, 1],
        base_channels=64,
        conv1_kernel=(1, 7, 7),
        inflate=(0, 0, 1, 1)),
    pose_pathway=dict(
        num_stages=3,
        stage_blocks=(4, 6, 3),
        lateral=True,
        lateral_inv=True,
        lateral_infl=16,
        lateral_activate=(0, 1, 1),
        in_channels=17,
        base_channels=32,
        out_indices=(2, ),
        conv1_kernel=(1, 7, 7),
        conv1_stride=(1, 1),
        pool1_stride=(1, 1),
        inflate=(0, 1, 1),
        spatial_strides=(2, 2, 2),
        temporal_strides=(1, 1, 1)))
head_cfg = dict(
    type='RGBPoseHead',
    num_classes=60,
    in_channels=[2048, 512],
    loss_components=['rgb', 'pose'],
    loss_weights=[1., 1.])
#test_cfg=dict(average_clips='prob', feat_ext=True)
model = dict(
    type='MMRecognizer3D',
    backbone=backbone_cfg,
    cls_head=head_cfg,
    test_cfg=dict(average_clips='prob', feat_ext=True))
memcached = True
mc_cfg = ('localhost', 22077)
dataset_type = 'PoseDataset'
data_root = r"C:\\Users\\emanuelefrassi\\OneDrive - Delft University of Technology\\Work\\Anomaly detection\\Videos_10s_processed\\{folder_name}\"
ann_file = r"{ann_file_path}"
left_kp=[1, 3, 5, 7, 9, 11, 13, 15]
right_kp=[2, 4, 6, 8, 10, 12, 14, 16]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
train_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
val_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=8, Pose=32), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
test_pipeline = [
    dict(type='MMUniformSampleFrames', clip_len=dict(RGB=50, Pose=200), num_clips=1),
    dict(type='MMDecode'),
    dict(type='MMCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(256, 256), keep_ratio=False),
    dict(type='GeneratePoseTarget', sigma=0.7, use_score=True, with_kp=True, with_limb=False, scaling=0.25),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['imgs', 'heatmap_imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'heatmap_imgs', 'label'])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=0,
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, split='xsub_train', data_prefix=data_root, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', data_prefix=data_root, pipeline=val_pipeline),
    test=dict(type=dataset_type, ann_file=ann_file, split='xsub_val', data_prefix=data_root, pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.0075, momentum=0.9, weight_decay=0.0001)  # this lr is used for 8 gpus
optimizer_config = dict(grad_clip=dict(max_norm=40, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[12, 16])
total_epochs = 20
checkpoint_config = dict(interval=1)
workflow = [('train', 1)]
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5), key_indicator='RGBPose_1:1_top1_acc')
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/rgbpose_conv3d/rgbpose_conv3d'
load_from = 'https://download.openmmlab.com/mmaction/pyskl/ckpt/rgbpose_conv3d/rgbpose_conv3d_init.pth'

"""

# Iterate over all .pkl files and create configurations
for file_name in os.listdir(pkl_dir):
    if file_name.endswith(".pkl"):
        pkl_path = os.path.join(pkl_dir, file_name)
        config_path = os.path.join(config_dir, f"config_{file_name}.py")
        folder_name_=os.path.split(file_name)[1].split(".")[0]
       # print(folder_name_)
        # Generate config content with updated ann_file path
        config_content = config_template.format(ann_file_path=pkl_path,folder_name=folder_name_)
        
        # Save the configuration file
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)

print(f"Configuration files created successfully. {config_path}")
