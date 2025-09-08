# Copyright (c) OpenMMLab. All rights reserved.
# flake8: noqa: E722
import argparse
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
from mmcv import Config
from mmcv import digit_version as dv
from mmcv import load
from mmcv.cnn import fuse_conv_bn
from mmcv.engine import multi_gpu_test,single_gpu_test
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from pyskl.datasets import build_dataloader, build_dataset
from pyskl.models import build_model
from pyskl.utils import cache_checkpoint, mc_off, mc_on, test_port
import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='pyskl test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('-C', '--checkpoint', help='checkpoint file', default=None)
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default=['top_k_accuracy', 'mean_class_accuracy'],
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple workers')
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'slurm'],
        default='pytorch',
        help='job launcher')
    parser.add_argument(
        '--compile',
        action='store_true',
        help='whether to compile the model before training / testing (only available in pytorch 2.0)')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--local-rank', type=int, default=-1)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def inference_pytorch_old(args, cfg, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        # You can set average_clips during testing, it will override the
        # original setting
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # build the model and load checkpoint
    model = build_model(cfg.model)

    if args.checkpoint is None:
        work_dir = cfg.work_dir
        args.checkpoint = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint)

    args.checkpoint = cache_checkpoint(args.checkpoint)
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    return outputs
def inference_pytorch(args, cfg, data_loader):
    """Get predictions by pytorch models."""
    if args.average_clips is not None:
        if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
            cfg.model.setdefault('test_cfg',
                                 dict(average_clips=args.average_clips))
        else:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # build the model and load checkpoint
    model = build_model(cfg.model)
   # import inspect
    #print(f"DEBUG: Source file of MMRecognizer3D (or its base): {inspect.getfile(type(model))}")
    
    if dv(torch.__version__) >= dv('2.0.0') and args.compile:
        model = torch.compile(model)

    if args.checkpoint is None:
        work_dir = cfg.work_dir
        args.checkpoint = osp.join(work_dir, 'latest.pth')
        assert osp.exists(args.checkpoint)

    args.checkpoint = cache_checkpoint(args.checkpoint)
 #   load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    # Single-GPU model execution
    from mmcv.parallel import MMDataParallel
    model = MMDataParallel(model.cuda(), device_ids=[0])
    
   # outputs = multi_gpu_test(model, data_loader, args.tmpdir)
    

    outputs = single_gpu_test(model, data_loader)
    #rgb_feat, pose_feat = single_gpu_test(model, data_loader)

    return outputs
    #return  rgb_feat, pose_feat



config_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Stick Figures\Stick Figures RDG\WallS\Final procedures\Pyskl_data\\configuration_rgbpose"

config_files = glob.glob(osp.join(config_dir, "config_*.py"))


for config_path in config_files:
    class Args:
        config = config_path
        checkpoint = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Checkpoint\checkpoint.pth"  
        out_dir = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Features\rgbpose"  
        eval = ["top_k_accuracy", "mean_class_accuracy"]  
        tmpdir = None
        average_clips = None
        launcher = "pytorch"
        compile = False
        local_rank = -1
        fuse_conv_bn = True
    failed_log_path = osp.join(Args.out_dir, "failed_files.txt")

    config_name = osp.basename(config_path).replace("config_", "").replace(".pkl.py", "")
    out_file = osp.join(Args.out_dir, f"{config_name}.pkl")
    
    if osp.exists(out_file):
        print(f"Skipping {config_name}, output already exists at {out_file}")
        continue

    mmcv.mkdir_or_exist(osp.dirname(out_file))

    try:
        cfg = Config.fromfile(config_path)

        eval_cfg = cfg.get("evaluation", {})
        for key in ["interval", "tmpdir", "start", "save_best", "rule", "by_epoch", "broadcast_bn_buffers"]:
            eval_cfg.pop(key, None)
        if Args.eval:
            eval_cfg["metrics"] = Args.eval

        cfg.data.test.test_mode = True
        if not hasattr(cfg, "dist_params"):
            cfg.dist_params = dict(backend="nccl")

        dataset = build_dataset(cfg.data.test, dict(test_mode=True))
        dataloader_setting = dict(
            workers_per_gpu=cfg.data.get("workers_per_gpu", 1),
            shuffle=False,
            **cfg.data.get("test_dataloader", {}),
        )
        data_loader = build_dataloader(dataset, **dataloader_setting)

        outputs = inference_pytorch(Args, cfg, data_loader)

        mmcv.dump(outputs, out_file)
        print(f"Saved results to {out_file}")

    except Exception as e:
         error_message = f"Failed config: {config_name}\nError: {repr(e)}\n\n"
         print(f"❌ {error_message}")
         with open(failed_log_path, "a") as f:
             f.write(error_message)

r'''
#%%
class Args:
    config = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Stick Figures\Codes\configuration_file.py"  # Path to config file
    checkpoint = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\Checkpoint\checkpoint.pth"  # Path to checkpoint file
    out = r"C:\Users\emanuelefrassi\OneDrive - Delft University of Technology\Work\Anomaly detection\Output\result.pkl"  # Output file path
    eval = ["top_k_accuracy", "mean_class_accuracy"]  # Evaluation metrics
    tmpdir = None
    average_clips = None
    launcher = "pytorch"
    compile = False
    local_rank = -1
    fuse_conv_bn=True

args = Args()
cfg = Config.fromfile(args.config)

out = osp.join(cfg.work_dir, 'result.pkl') if args.out is None else args.out

# Load eval_config from cfg
eval_cfg = cfg.get('evaluation', {})
keys = ['interval', 'tmpdir', 'start', 'save_best', 'rule', 'by_epoch', 'broadcast_bn_buffers']
for key in keys:
    eval_cfg.pop(key, None)
if args.eval:
    eval_cfg['metrics'] = args.eval

mmcv.mkdir_or_exist(osp.dirname(out))
_, suffix = osp.splitext(out)
assert suffix[1:] in file_handlers, ('The format of the output file should be json, pickle or yaml')

# set cudnn benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True
cfg.data.test.test_mode = True

if not hasattr(cfg, 'dist_params'):
    cfg.dist_params = dict(backend='nccl')

#init_dist(args.launcher, **cfg.dist_params)
#rank, world_size = get_dist_info()
#cfg.gpu_ids = range(world_size)

# build the dataloaderq
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
dataloader_setting = dict(
    videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
    workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
    shuffle=False)
dataloader_setting = dict(dataloader_setting, **cfg.data.get('test_dataloader', {}))
data_loader = build_dataloader(dataset, **dataloader_setting)

default_mc_cfg = ('localhost', 22077)
memcached = cfg.get('memcached', False)


outputs = inference_pytorch(args, cfg, data_loader)
if rank == 0 and memcached:
    # mc_list is a list of pickle files you want to cache in memory.
    # Basically, each pickle file is a dictionary.
    mc_cfg = cfg.get('mc_cfg', default_mc_cfg)
    assert isinstance(mc_cfg, tuple) and mc_cfg[0] == 'localhost'
    if not test_port(mc_cfg[0], mc_cfg[1]):
        mc_on(port=mc_cfg[1], launcher=args.launcher)
    retry = 3
    while not test_port(mc_cfg[0], mc_cfg[1]) and retry > 0:
        time.sleep(5)
        retry -= 1
    assert retry >= 0, 'Failed to launch memcached. '

dist.barrier()

rank, _ = get_dist_info()
if rank == 0:
    print(f'\nwriting results to {out}')
    dataset.dump_results(outputs, out=out)
    if eval_cfg:
        eval_res = dataset.evaluate(outputs, **eval_cfg)
        for name, val in eval_res.items():
            print(f'{name}: {val:.04f}')

dist.barrier()
if rank == 0 and memcached:
    mc_off()

'''
