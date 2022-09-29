import argparse
import os
import sys
import uuid
from datetime import datetime as dt

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from thop import profile
from tqdm import tqdm

import model_io
import utils
import wandb
from dataloader import DepthDataLoader
from loss import SILogLoss
# Models
from models.EFT import EFT
from models.EFT_v2 import EFT_v2
from models.midas.midas_net_custom import MidasNet_small
from utils import RunningAverage, colorize, send_massage

# os.environ['WANDB_MODE'] = 'dryrun'
PROJECT = "depthflow"
logging = True
token = "c9749a54b69e"

def is_rank_zero(args):
    return args.rank == 0

def log_images(img, depth, pred, args, step):
    depth = colorize(depth, vmin=args.min_depth, vmax=args.max_depth)
    pred = colorize(pred, vmin=args.min_depth, vmax=args.max_depth)
    wandb.log(
        {
            "_Input": [wandb.Image(img)],
            "_GT": [wandb.Image(depth)],
            "_Prediction": [wandb.Image(pred)]
        }, step=step)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    ###################################### Load model ##############################################
    model = EFT_v2()
    ################################################################################################

    if args.gpu is not None:  # If a gpu is set by user: NO PARALLELISM!!
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    args.multigpu = False
    if args.distributed:
        # Use DDP
        args.multigpu = True
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        # args.batch_size = 8
        args.workers = int((args.num_workers + ngpus_per_node - 1) / ngpus_per_node)
        print("gpu: {} rank: {} batch_size: {} workers: {}".format(args.gpu, args.rank, args.batch_size, args.workers))
        torch.cuda.set_device(args.gpu)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                          find_unused_parameters=True)

    elif args.gpu is None:
        # Use DP
        args.multigpu = True
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    args.epoch = 0
    args.last_epoch = -1
    train(model, args, epochs=args.epochs, lr=args.lr, device=args.gpu, root=args.root,
          experiment_name=args.name, optimizer_state_dict=None)


def train(model, args, epochs=10, experiment_name="DeepLab", lr=0.0001, root=".", device=None,
          optimizer_state_dict=None):
    global PROJECT
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    ###################################### Logging setup #########################################
    print(f"Training {experiment_name}")

    # run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}-{uuid.uuid4()}"
    run_id = f"{dt.now().strftime('%d-%h_%H-%M')}-nodebs{args.bs}-tep{epochs}-lr{lr}-wd{args.wd}"
    name = f"{experiment_name}_{run_id}"
    should_write = ((not args.distributed) or args.rank == 0)
    should_log = should_write and logging
    if should_log:
        tags = args.tags.split(',') if args.tags != '' else None
        if args.dataset != 'nyu':
            PROJECT = PROJECT + f"-{args.dataset}"
        wandb.init(project=PROJECT, name=name, config=args, dir=args.root, tags=tags, notes=args.notes)
        wandb.watch(model)
    ################################################################################################

    train_loader = DepthDataLoader(args, 'train').data
    test_loader = DepthDataLoader(args, 'online_eval').data

    ###################################### losses ##############################################
    silog_loss = SILogLoss()
    ################################################################################################

    model.train()  # trun to train mode

    ###################################### Optimizer ################################################
    if args.same_lr:  # 学习率不变
        print("Using same LR")
        params = model.parameters()  # 返回该模型的参数
    else:  # 变化的学习率
        print("Using diff LR")
        m = model.module if args.multigpu else model
        params = [{"params": m.get_1x_lr_params(), "lr": lr / 10},
                  {"params": m.get_10x_lr_params(), "lr": lr}]

    optimizer = optim.AdamW(params, weight_decay=args.wd, lr=args.lr)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    ################################################################################################
    # some globals
    iters = len(train_loader)
    step = args.epoch * iters
    best_loss = np.inf

    ###################################### Scheduler ###############################################
    # 学习率按照一定的曲线变化
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=0.85, max_momentum=0.95, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    # import params from checkpoints
    if args.resume != '' and scheduler is not None:
        model, optimizer, args.epoch, step, best_loss = model_io.load_checkpoint(args.resume, model, optimizer)
        scheduler.step(args.epoch + 1)
    ################################################################################################

    # max_iter = len(train_loader) * epochs
    for epoch in range(args.epoch, epochs):
        ################################# Train loop ##########################################################
        if should_log: wandb.log({"Epoch": epoch}, step=step)
        for i, batch in tqdm(enumerate(train_loader), desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Train",
                             total=len(train_loader)) if is_rank_zero(
                args) else enumerate(train_loader):

            optimizer.zero_grad()  # 每个迭代步的梯度初始化为0

            img = batch['image'].to(device)  # B, 3, H, W
            depth = batch['depth'].to(device)  # B, 1, H, W
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue

            pred = model(img)  # (bx1xhxw)

            mask = depth > args.min_depth
            train_loss = silog_loss(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            
            train_loss.backward()  # 损失的后向传播，计算梯度
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # 可选，用于梯度剪裁
            optimizer.step()  # 使用梯度进行优化
            if should_log and step % 5 == 0:
                wandb.log({f"Train/{silog_loss.name}": train_loss.item()}, step=step)

            step += 1
            scheduler.step()  # 调整LR

            ########################################################################################################

            if should_write and step % args.validate_every == 0:

                ################################# Validation loop ##################################################
                model.eval()
                metrics, val_si = validate(args, model, test_loader, silog_loss, epoch, epochs, device)
                
                if should_log:
                    wandb.log({
                        f"Test/{silog_loss.name}": val_si.get_value(),
                    }, step=step)

                    wandb.log({f"Metrics/{k}": v for k, v in metrics.items()}, step=step)
                    model_io.save_checkpoint(model, optimizer, epoch, best_loss, f"{experiment_name}_{run_id}_latest.pt",
                                             root=os.path.join(root, "checkpoints"))
                    model_io.save_weights(model, f"{experiment_name}_{run_id}_latest.pt")
                    # Draw Picture
                    log_images(img, depth, pred, args, step)

                if metrics['abs_rel'] < best_loss and should_write:
                    model_io.save_checkpoint(model, optimizer, epoch,best_loss, f"{experiment_name}_{run_id}_best.pt",
                                             root=os.path.join(root, "checkpoints"))
                    model_io.save_weights(model, f"{experiment_name}_{run_id}_best.pt")
                    best_loss = metrics['abs_rel']
                model.train()
                #################################################################################################
        
        # Send Message to WeChat
        message = dict(epoch=epoch, a1=metrics['a1'], a2=metrics['a2'], a3=metrics['a3'], 
                    abs_rel=metrics['abs_rel'],rmse=metrics['rmse'], log_10=metrics['log_10'],
                    rmse_log=metrics['rmse_log'],silog=metrics['silog'], sq_rel=metrics['sq_rel'])
        send_massage(token, PROJECT, "#2", message)
    

    return model


def validate(args, model, test_loader, loss_function, epoch, epochs, device='cpu'):
    with torch.no_grad():
        val_si = RunningAverage()
        metrics = utils.RunningAverageDict()
        for batch in tqdm(test_loader, desc=f"Epoch: {epoch + 1}/{epochs}. Loop: Validation") if is_rank_zero(
                args) else test_loader:
            img = batch['image'].to(device)
            depth = batch['depth'].to(device)
            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    continue
            depth = depth.squeeze().unsqueeze(0).unsqueeze(0)
            pred = model(img)

            mask = depth > args.min_depth
            l_dense = loss_function(pred, depth, mask=mask.to(torch.bool), interpolate=True)
            val_si.append(l_dense.item())

            pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)

            pred = pred.squeeze().cpu().numpy()
            pred[pred < args.min_depth_eval] = args.min_depth_eval
            pred[pred > args.max_depth_eval] = args.max_depth_eval
            pred[np.isinf(pred)] = args.max_depth_eval
            pred[np.isnan(pred)] = args.min_depth_eval

            gt_depth = depth.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt_depth.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
            valid_mask = np.logical_and(valid_mask, eval_mask)
            metrics.update(utils.compute_errors(gt_depth[valid_mask], pred[valid_mask]))

        return metrics.get_value(), val_si


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Training script. Default values of all arguments are recommended for reproducibility', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args

    # Train Args
    parser.add_argument('--model', '-m', default=None, type=str, help='')
    parser.add_argument('--epochs', default=25, type=int, help='number of total epochs to run')
    # parser.add_argument('--n-bins', '--n_bins', default=80, type=int,
                        # help='number of bins/buckets to divide depth range into')
    parser.add_argument('--lr', '--learning-rate', default=0.000357, type=float, help='max learning rate')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float, help='weight decay')  # 权值衰减
    parser.add_argument('--w_chamfer', '--w-chamfer', default=0.1, type=float, help="weight value for chamfer loss")
    parser.add_argument('--div-factor', '--div_factor', default=25, type=float, help="Initial div factor for lr")
    parser.add_argument('--final-div-factor', '--final_div_factor', default=100, type=float,
                        help="final div factor for lr")

    parser.add_argument('--bs', default=16, type=int, help='batch size')
    parser.add_argument('--validate-every', '--validate_every', default=100, type=int, help='validation period')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument("--name", default="Demo", type=str)
    parser.add_argument("--norm", default="linear", type=str, help="Type of norm/competition for bin-widths",
                        choices=['linear', 'softmax', 'sigmoid'])
    parser.add_argument("--same-lr", '--same_lr', default=False, action="store_true",
                        help="Use same LR for all param groups")
    parser.add_argument("--distributed", default=False, action="store_true", help="Use DDP if set")
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")
    parser.add_argument("--resume", default='', type=str, help="Resume from checkpoint")

    parser.add_argument("--notes", default='', type=str, help="Wandb notes")
    parser.add_argument("--tags", default='sweep', type=str, help="Wandb tags")

    parser.add_argument("--workers", default=11, type=int, help="Number of workers for data loading")

    # Dataset Args
    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='dataset/nyu/sync/', type=str,
                        help="path to dataset")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')
    # 用于随机剪裁
    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)
    parser.add_argument('--random_crop_ratio', type=float, help='random crop ratio', default=None)

    parser.add_argument('--do_random_rotate', default=True,
                        help='if set, will perform random rotation for augmentation',
                        action='store_true')
    parser.add_argument('--degree', type=float, help='random rotation maximum degree', default=2.5)
    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
    parser.add_argument('--use_right', help='if set, will randomly use right images when train on KITTI',
                        action='store_true')

    parser.add_argument('--data_path_eval',
                        default="dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', default=True, help='if set, crops according to Eigen NIPS14',
                        action='store_true')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()  # 建立解析对象 args

    args.batch_size = args.bs
    args.num_threads = args.workers
    args.mode = 'train'
    args.chamfer = args.w_chamfer > 0  #! 倒角损失
    if args.root != "." and not os.path.isdir(args.root):
        os.makedirs(args.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace('[', '').replace(']', '')
        nodes = node_str.split(',')

        args.world_size = len(nodes)
        args.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        args.world_size = 1
        args.rank = 0
        nodes = ["127.0.0.1"]

    if args.distributed:
        mp.set_start_method('forkserver')

        print("rank: {}".format(args.rank))
        port = np.random.randint(15000, 15025)
        args.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        print("dist_url: {}".format(args.dist_url))
        args.dist_backend = 'nccl'
        args.gpu = None

    ngpus_per_node = torch.cuda.device_count()
    args.num_workers = args.workers
    args.ngpus_per_node = ngpus_per_node  # Number of Gpus

    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        if ngpus_per_node == 1:
            args.gpu = 0
        main_worker(args.gpu, ngpus_per_node, args)
