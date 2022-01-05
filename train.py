import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

import segmentation_models_pytorch as smp
from torch.optim.swa_utils import AveragedModel, SWALR

import argparse
import wandb
import os
import random
import sys
import time
from pathlib import Path
from tqdm import tqdm
import argparse

from datasets import HeartDataset
from losses import *
from utils import *

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_aug(arg):
    return A.Compose([
        A.Affine(),
        A.GaussNoise(),
        A.Resize(height=arg.img_size, width=arg.img_size, p=1.0),
        A.Normalize(mean=(0.5,), std=(0.5,), p = 1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)


def valid_aug(arg):
    return A.Compose(
        [
            A.Resize(height=arg.img_size, width=arg.img_size, p=1.0),
            A.Normalize(mean=(0.5,), std=(0.5, ), p=1.0),
            ToTensorV2(p=1.0),
        ], 
        p=1.0
    )

#def train(train_loader, model, optimizer, loss_fn, device, epoch):
def train(train_loader, model, optimizer, scheduler, loss_fn, device, epoch, 
          swa_mode, swa_start ,swa_model = None, swa_scheduler=None):
    meter = Meter("train", epoch)
    start = time.strftime("%H:%M:%S")

    model.train()
    pbar = tqdm(enumerate(train_loader), total = len(train_loader))
    running_loss = 0.0
    total_batches = len(train_loader)

    for batch_idx, (img, mask) in pbar:
        images = img.to(device).float()
        masks = mask.to(device).float()
        
        pred = model(images)
        loss = loss_fn(pred, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        running_loss += loss.item()
        pred = pred.detach().cpu()
        
        if not swa_mode:
            scheduler.step()
        else:
            if epoch <= swa_start:
                scheduler.step()

        meter.update(mask, pred)
        pbar.set_postfix(epoch = epoch, loss = running_loss / (batch_idx + 1))

    if epoch > swa_start and swa_mode:
        swa_model.update_parameters(model)
        swa_scheduler.step()

    epoch_loss = (running_loss) / total_batches
    dice, iou = epoch_log("train", epoch, epoch_loss, meter, start)
    wandb.log({"Train Dice" : dice ,"Train IoU" : iou, "Train loss" : epoch_loss})
    return epoch_loss, dice, iou

def valid(valid_loader, model, loss_fn, device, epoch):
    meter = Meter("valid", epoch)
    model.eval()
    start = time.strftime("%H:%M:%S")
    pbar = tqdm(enumerate(valid_loader), total = len(valid_loader))
    running_loss = 0.0
    total_batches = len(valid_loader)

    valid_gt = []
    valid_pred = []

    with torch.no_grad():
        for batch_idx, (img, mask) in pbar:
            imgs = img.to(device).float()
            masks = mask.to(device).float()

            pred = model(imgs)
            loss = loss_fn(pred, masks)

            running_loss += loss.item()
            pred = pred.detach().cpu()
            
            valid_gt.append(wandb.Image(masks[0], caption="Valid GT"))
            valid_pred.append(wandb.Image(pred[0], caption="Valid Predictions"))
            meter.update(mask, pred)
            pbar.set_postfix(epoch = epoch, loss = running_loss / (batch_idx + 1))

    epoch_loss = (running_loss) / total_batches
    dice, iou = epoch_log("valid", epoch, epoch_loss, meter, start)
    wandb.log({"Valid Dice" : dice ,"valid IoU" : iou, "Valid loss" : epoch_loss, "Valid GT" : valid_gt, "valid_pred" : valid_pred})
    return epoch_loss, dice, iou

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--swa_lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=6, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--img_size', type=int, default=416, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--swa', action='store_true', help='use Stochastic Weight Averaging')
    parser.add_argument('--swa_start', type=int, default=20, help='swa start per epochs')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='use torch.optim.AdamW() optimizer with weight_decay')
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--backbone', metavar='ARCH', default='resnet34',
                        help='model backbone')
    parser.add_argument('--save_period', type=int, default=-1,
                        help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--num_comb', type=int, default=-1,
                        help='number of combination about augmentation')
    parser.add_argument('--name', type=str, default='base',
                        help='project name')                  
    parser.add_argument('--input_path', default='dataset',
                        help='input data path')             
    parser.add_argument('--domain', default='all',
                        help='all / A2C / A4C')                                     
    return parser

def main():

    set_seed(42)
    opt = parse_opt()
    args = opt.parse_args()

    model_name = args.name

    wandb.init(project="heart-disease", group='resnet_4ch', name=model_name, reinit=True)
    wandb.config.update(args)

    gpu_devices = ','.join([str(id) for id in args.device])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #dataset
    train_dataset = HeartDataset(args.input_path, mode = "train", domain=args.domain, transforms = train_aug(arg = args))
    valid_dataset = HeartDataset(args.input_path, mode = "validation", domain=args.domain, transforms = valid_aug(arg = args))
    if train_dataset[0] and valid_dataset[0]:
        print("Dataset Loaded")
    else:
        quit()
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size , shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size , shuffle=False, num_workers=4, pin_memory=True)
    
    criterion = DiceLoss()
    criterion = criterion.to(device)
    model = smp.UnetPlusPlus(args.backbone, encoder_weights="imagenet", activation=None, in_channels=4)
    model.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model = model.to(device)
    

    # for optimizer
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr)

    # for scheduler
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.min_lr)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.factor, patience=args.patience, verbose=1, min_lr=args.min_lr)
    elif args.scheduler == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in args.milestones.split(',')], gamma=args.gamma)
    elif args.scheduler == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    swa_model = AveragedModel(model)
    swa_model = swa_model.to(device)
    swa_scheduler = SWALR(optimizer, swa_lr=args.swa_lr)
    best_loss = 0.5
    best_dice = 0.0
    chk = 0
    wandb.watch(model)

    for epoch in range(args.epochs):
        print(f"############################### EPOCH : {epoch+1} ######################################")
        # train_ep_loss, train_dice, train_iou = train(train_loader, model, optimizer, loss_fn = criterion, device=device, epoch=epoch)
        train_ep_loss, train_dice, train_iou = train(train_loader, model, optimizer, scheduler, loss_fn = criterion, device=device, epoch=epoch,
                                                    swa_start = args.swa_start ,swa_mode=args.swa, swa_model = swa_model, swa_scheduler = swa_scheduler)
        valid_ep_loss, valid_dice, valid_iou = valid(valid_loader, model, loss_fn = criterion, device=device, epoch=epoch)
        scheduler.step()
        
        if valid_dice < best_dice:
            chk += 1
        if chk == args.early_stopping:
            break
        else:
            best_dice = valid_dice
            chk = 0

        if valid_ep_loss < best_loss:
            checkpoint = {
                "state_dict" : model.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_loss" : best_loss,
                "epoch" : epoch
            }

            best_loss = valid_ep_loss
            print('epoch : {} best_loss {}'.format(epoch, best_loss))

            torch.save(checkpoint, f"./checkpoint/model_best_{model_name}.pth")

        if epoch == (args.epochs - 1) and args.swa:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device = device)
            checkpoint = {
                "state_dict" : swa_model.module.state_dict(),
                "optimizer" : optimizer.state_dict(),
                "best_loss" : valid_ep_loss,
                "epoch" : epoch
            }

            torch.save(checkpoint, f"./checkpoint/model_final_{model_name}.pth")

if __name__ =="__main__":
    main()
