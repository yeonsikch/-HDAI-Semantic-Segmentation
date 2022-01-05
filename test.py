import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

from datasets import HeartDataset
from losses import *
from utils import *
from train import valid_aug

import torch
import numpy as np
import matplotlib.pyplot as plt

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

plt.rcParams["savefig.bbox"] = 'tight'

chk = 0

def test(test_loader, model, device):
    model.eval()
    pbar = tqdm(enumerate(test_loader), total = len(test_loader))

    test_pred, mask_list, width_list, height_list = [], [], [], []

    with torch.no_grad():
        for batch_idx, value in pbar:
            img, mask, height, width = value
            imgs = img.to(device).float()
            pred = torch.sigmoid(model(imgs))
            pred[pred>=0.5]=1
            pred[pred<0.5]=0
            pred = pred.detach().cpu()
            test_pred.extend(pred.squeeze())
            mask_list.extend(mask)
            height_list.extend(height)
            width_list.extend(width)

    return test_pred, mask_list, height_list, width_list

def calc_metric(pred, mask, height, width):
    pred = T.Resize([height,width])(pred.unsqueeze(0)).squeeze()
    mask = T.Resize([height,width])(mask.unsqueeze(0)).squeeze()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0
    tp = (pred*mask).sum()
    fp = ((pred-mask)==1).sum()
    fn = ((mask-pred)==1).sum()
    dice = 2*tp/(2*tp+fp+fn)
    ji = dice/(2-dice)
    return dice.item(), ji.item()

def save_npy(pred, height, width, idx, save_path):
    pred = T.Resize([height,width])(pred.unsqueeze(0)).squeeze()
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    pred = pred.numpy()
    filename = os.path.join(save_path, f'{str(idx+1).zfill(4)}.npy')
    np.save(filename, pred)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=6, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--img_size', type=int, default=416, help='train, val image size (pixels)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--backbone', action='append',
                        help='model backbone')
    parser.add_argument('--weight', action='append', 
                        help='load weight path')                    
    parser.add_argument('--input_path', default='dataset',
                        help='input data path')             
    parser.add_argument('--domain', default='all',
                        help='all / A2C / A4C')                                     
    return parser

def main():
    opt = parse_opt()
    args = opt.parse_args()

    gpu_devices = ','.join([str(id) for id in args.device])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #dataset
    test_dataset = HeartDataset(args.input_path, mode = "test", domain=args.domain, transforms = valid_aug(arg = args))
    if len(test_dataset):
        print("Dataset Loaded")
    else:
        quit()
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size , shuffle=False, num_workers=4, pin_memory=True)

    if type(args.weight) == 'str':
        args.weight = [args.weight]

    test_pred = torch.zeros((len(test_dataset), args.img_size, args.img_size))

    print('Number of ensemble :', len(args.weight))
    
    for backbone, weight in zip(args.backbone, args.weight):
        # model = smp.UnetPlusPlus(backbone, encoder_weights="imagenet", activation=None, in_channels=4)
        model = smp.UnetPlusPlus(backbone, encoder_weights="imagenet", activation=None, in_channels=4)
        model.encoder.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.load_state_dict(torch.load(weight, map_location=device)['state_dict'])
        model = model.to(device)
        pred, mask_list, height_list, width_list = test(test_loader, model, device=device)
        test_pred += torch.stack(pred)
    
    test_pred = test_pred / len(args.weight)
    test_pred[test_pred>=0.5] = 1
    test_pred[test_pred<0.5] = 0

    save_path = f'./detection-results/{args.domain.upper()}'
    os.makedirs(save_path, exist_ok=True)
    dice_score, ji_score = 0, 0
    for i in range(len(test_pred)):
        save_npy(test_pred[i], height_list[i], width_list[i], i, save_path)
        score = calc_metric(test_pred[i], mask_list[i], height_list[i], width_list[i])
        dice_score += score[0]
        ji_score += score[1]

    dice_score /= len(test_pred)
    ji_score /= len(test_pred)
    print('DICE SCORE :', dice_score)
    print('JI SCORE :', ji_score)
    print('Mean Score :', (dice_score+ji_score)/2)

if __name__ =="__main__":
    main()
