import argparse
import logging
import math
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

from utils.loss import ComputeLoss
from torch.utils.tensorboard import SummaryWriter

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
LOGGER = logging.getLogger(__name__)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from utils.general import *
from utils.torch_utils import *
from utils.callbacks import Callbacks
from models.yolo import Model
from dataset.pro_dataset import Whu_dataset, Whu_sub_dataset
from torch.utils.data import random_split, DataLoader

val_loader = []


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'last.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='models/yolov5l.yaml', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/whu_rsipac.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=3, help='total batch siSze for all GPUs')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=12, help='maximum number of dataloader workers')
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', type=int, default=0, help='Number of layers to freeze. backbone=10, all=24')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    # Weights & Biases arguments
    parser.add_argument('--entity', default=None, help='W&B: Entity')
    parser.add_argument('--upload_dataset', action='store_true', help='W&B: Upload dataset as artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='W&B: Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='W&B: Version of dataset artifact to use')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls = Path(
        opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls
    evolve, data, cfg, resume, noval = opt.evolve, opt.data, opt.cfg, opt.resume, opt.noval
    nosave, workers, freeze = opt.nosave, opt.workers, opt.freeze

    # Directories
    w = save_dir / 'weights'
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)

    # Save run settings
    with open(save_dir / 'hyp.yaml', 'w') as f:
        yaml.safe_dump(hyp, f, sort_keys=False)
    with open(save_dir / 'opt.yaml', 'w') as f:
        yaml.safe_dump(vars(opt), f, sort_keys=False)
    with open(opt.data, encoding='UTF-8', errors='ignore') as f:
        data_dict = yaml.safe_load(f)  # 加载数据集路径
    # Config
    plots = not evolve
    cuda = device.type != 'cpu'
    init_seeds(0)
    dataset_root_path = data_dict['path']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == nc, f'{len(names)} names found for nc={nc} dataset in {data}'  # check
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        MULTI_GPU = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        device_ids = [x for x in range(torch.cuda.device_count())]
        count_str = ''
        for i in device_ids:
            count_str += str(i)
            count_str += ','
        os.environ["CUDA_VISIBLE_DEVICES"] = count_str
    else:
        MULTI_GPU = False
    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    if pretrained:
        weights = attempt_download(weights)  # download if not found locally
        ckpt = torch.load(weights, map_location=device)  # load checkpoint
        model = Model(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        if MULTI_GPU:
            model = nn.DataParallel(model, device_ids=device_ids)
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        if MULTI_GPU:
            model = nn.DataParallel(model, device_ids=device_ids)

    # Freeze
    freeze = [f'model.{x}.' for x in range(freeze)]  # layers to freeze
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print(f'freezing {k}')
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay

    g0, g1, g2 = [], [], []  # optimizer parameter groups
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):  # bias
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):  # weight (no decay)
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g1.append(v.weight)

    if opt.adam:
        optimizer = Adam(g0, lr=hyp['lr0'], betas=(hyp['momentum'], 0.999))  # adjust beta1 to momentum
    else:
        optimizer = SGD(g0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': g1, 'weight_decay': hyp['weight_decay']})  # add g1 with weight_decay
    optimizer.add_param_group({'params': g2})  # add g2 (biases)
    del g0, g1, g2
    #   Scheduler
    if opt.linear_lr:
        lf = lambda x: (1 - x / (epochs - 1)) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
    else:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)
    if MULTI_GPU:
        optimizer = nn.DataParallel(optimizer, device_ids=device_ids)
        scheduler = nn.DataParallel(scheduler, device_ids=device_ids)
    # ema
    ema = ModelEMA(model)
    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        # Optimizer
        if ckpt['optimizer'] is not None:
            optimizer.module.load_state_dict(ckpt['optimizer']) if MULTI_GPU else optimizer.load_state_dict(
                ckpt['optimizer'])
            best_fitness = ckpt['best_fitness']

        # EMA
        if ema and ckpt.get('ema'):
            ema.ema.load_state_dict(ckpt['ema'].float().state_dict())
            ema.updates = ckpt['updates']

        # Epochs
        start_epoch = ckpt['epoch'] + 1
        if resume:
            assert start_epoch > 0, f'{weights} training to {epochs} epochs is finished, nothing to resume.'
        if epochs < start_epoch:
            epochs += ckpt['epoch']  # finetune additional epochs

        del ckpt, csd
    # Image sizes
    gs = max(int(model.module.stride.max()), 32) if MULTI_GPU else max(int(model.stride.max()), 32)
    # grid size (max stride)
    nl = model.module.model[-2].nl if MULTI_GPU else model.model[-2].nl
    # number of detection layers (used for scaling hyp['obj'])
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # DataLoader
    whu_dataset = Whu_dataset(dataset_root_path, cache=True)
    train_dataset_size = int(len(whu_dataset) * data_dict['split_percent'])
    train_dataset_indexs = np.random.choice(np.arange(len(whu_dataset)), train_dataset_size, replace=False)
    val_dataset_indexs = np.setdiff1d(np.arange(len(whu_dataset)), train_dataset_indexs)
    train_dataset = Whu_sub_dataset(dataset_root_path, train_dataset_indexs)
    val_dataset = Whu_sub_dataset(dataset_root_path, val_dataset_indexs)
    train_loader = DataLoader(train_dataset,
                              batch_size=opt.batch_size,
                              shuffle=False,
                              num_workers=opt.workers,
                              collate_fn=Whu_dataset.col_fun)
    val_loader = DataLoader(val_dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.workers,
                            collate_fn=Whu_dataset.col_fun)
    nb = len(train_loader)  # number of batches
    # Model parameters
    hyp['box'] *= 3. / nl
    hyp['cls'] *= nc / 80. * 3. / nl
    hyp['obj'] *= (imgsz / 640)**2 * 3. / nl
    hyp['label_smoothing'] = opt.label_smoothing
    if MULTI_GPU:
        model.module.nc = nc
        model.module.hyp = hyp
        model.module.class_weights = labels_to_class_weights_whu(train_dataset.lables, nc).to(device) * nc
        model.module.names = names
    else:
        model.nc = nc
        model.hyp = hyp
        model.class_weights = labels_to_class_weights_whu(train_dataset.lables, nc).to(device) * nc
        model.names = names
    # start training
    t0 = time.time()
    nw = max(round(hyp['warmup_epochs'] * nb), 1000)  # number of warmup iterations,
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)
    scaler = amp.GradScaler(enabled=cuda)
    stopper = EarlyStopping(patience=opt.patience)
    compute_loss = ComputeLoss(model, Multi_gpu=MULTI_GPU)
    with open(save_dir / 'result.txt', 'a+') as f:
        write_line = ('%-10s' * 6) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'seg') + '\n'
        f.writelines(write_line)
    f.close()
    print(('\n' + '%-10s' * 8) % ('Epoch', 'gpu_mem', 'box', 'obj', 'cls', 'seg', 'labels', 'img_size'))
    writter = SummaryWriter(save_dir)
    for epoch in range(start_epoch, epochs):  # epoch-----------------------
        gl1 = 5
        gl2 = 30
        gl3 = 30
        gl4 = 3
        gl_all = torch.tensor([gl1, gl2, gl3, gl4], device=device)
        model.train()
        mloss = torch.zeros(4, device=device)  # mean loss
        pbar = enumerate(train_loader)
        pbar = tqdm(pbar, total=nb)
        optimizer.module.zero_grad() if MULTI_GPU else optimizer.zero_grad()
        for i, (imgs, masks, lables) in pbar:  # batch
            n_lables = 0
            for lable in lables:
                if lable != 0:
                    n_lables += len(lable)
            # continue
            ni = i + nb * epoch
            imgs = imgs.to(device)
            # Warmup
            if ni < nw:
                xi = [0, nw]
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.module.param_groups if MULTI_GPU else optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, xi, [hyp['warmup_bias_lr'] if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, xi, [hyp['warmup_momentum'], hyp['momentum']])

            # Multi-scale
            # if opt.multi_scale:
            #     sz = random.randrange(imgsz * 0.5, imgsz * 1.5 + gs) // gs * gs  # size
            #     sf = sz / max(imgs.shape[2:])  # scale factor
            #     if sf != 1:
            #         ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
            #         imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            with amp.autocast(enabled=cuda):
                pred = model(imgs)
                loss, loss_items = compute_loss(pred, lables, masks, device=device)

            # Backward
            scaler.scale(loss).backward()

            # Optimize
            if ni - last_opt_step >= accumulate:
                scaler.step(optimizer.module if MULTI_GPU else optimizer)  # optimizer.step
                scaler.update()
                optimizer.module.zero_grad() if MULTI_GPU else optimizer.zero_grad()
                if ema:
                    ema.update(model.module if MULTI_GPU else model)
                last_opt_step = ni

            # Log
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)
            pbar.set_description(('%-10s' * 2 + '%-10.5f' * 4 + ('%-10d') * 2) %
                                 (f'{epoch}/{epochs - 1}', mem, *(mloss * gl_all), n_lables, imgs.shape[-1]))
        # Scheduler
        # lr = [x['lr'] for x in optimizer.module.param_groups]  # for loggers
        scheduler.module.step() if MULTI_GPU else scheduler.step()

        # Logggggg
        with open(save_dir / 'result.txt', 'a+') as f:
            write_line = ('%-10s' * 2 + '%-10.5f' * 4) % (f'{epoch}/{epochs - 1}', mem, *(mloss * gl_all)) + '\n'
            f.writelines(write_line)
        f.close()
        writter.add_scalar('lbox', mloss[0] * gl1, epoch + 1)
        writter.add_scalar('lobj', mloss[1] * gl2, epoch + 1)
        writter.add_scalar('lcls', mloss[2] * gl3, epoch + 1)
        writter.add_scalar('lseg', mloss[3] * gl4, epoch + 1)
        ema.update_attr(model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])
        final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
        # if not noval or final_epoch:  # Calculate mAP
        #     results, maps, _ = val.run(data_dict,
        #                                batch_size=batch_size // WORLD_SIZE * 2,
        #                                imgsz=imgsz,
        #                                model=ema.ema,
        #                                single_cls=single_cls,
        #                                dataloader=val_loader,
        #                                save_dir=save_dir,
        #                                plots=False,
        #                                callbacks=callbacks,
        #                                compute_loss=compute_loss)
        #Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
        if fi > best_fitness:
            best_fitness = fi

        # Save model
        if (not nosave) or (final_epoch and not evolve):  # if save
            ckpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'model': deepcopy(de_parallel(model)).half(),
                'ema': deepcopy(ema.ema).half(),
                'updates': ema.updates,
                'optimizer': optimizer.module.state_dict() if MULTI_GPU else optimizer.state_dict(),
            }

            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fi:
                torch.save(ckpt, best)
            if (epoch > 0) and (opt.save_period > 0) and (epoch % opt.save_period == 0):
                torch.save(ckpt, w / f'epoch{epoch}.pt')
            del ckpt
        for f in last, best:
            if f.exists():
                pass
                # strip_optimizer(f)  # strip optimizers
                # if f is best:
                #     results, _, _ = val.run(
                #         data_dict,
                #         batch_size=batch_size // WORLD_SIZE * 2,
                #         imgsz=imgsz,
                #         model=attempt_load(f, device).half(),
                #         iou_thres=0.65 if is_coco else 0.60,  # best pycocotools results at 0.65
                #         single_cls=single_cls,
                #         dataloader=val_loader,
                #         save_dir=save_dir,
                #         save_json=is_coco,
                #         verbose=True,
                #         plots=True,
                #         callbacks=callbacks,
                #         compute_loss=compute_loss)  # val best model with plots
    torch.cuda.empty_cache()
    return results


def main(opt, callbacks=Callbacks()):
    # 断点恢复
    if opt.resume:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(Path(ckpt).parent.parent / 'opt.yaml', errors='ignore') as f:
            opt = argparse.Namespace(**yaml.safe_load(f))  # replace
        opt.cfg, opt.weights, opt.resume = '', ckpt, True  # reinstate
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  #增量式路径

    # select device and ddp mode 因只有单卡 后续代码没有附上
    device = select_device(opt.device, batch_size=opt.batch_size)

    # 训练, 使用evolve代码没有加入

    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
