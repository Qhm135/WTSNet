from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from dataloader import myloader as DA
from models import *
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from err_calculation import *

parser = argparse.ArgumentParser(description='TANet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='max disp')
parser.add_argument('--dataset', default='kitti2012',
                    help='datapath')
parser.add_argument('--datapath', default='dataset/kitti2012_train/',
                    help='datapath')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default=None,
                    help='load model')
parser.add_argument('--savemodel', default='./ckpt/',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--logdir', default='log_dir',
                    help='save log')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.dataset == 'kitti2012':
    from dataloader import myloader12 as ls
    args.datapath = 'dataset/kitti2012_train/'
elif args.dataset == 'kitti2015':
    from dataloader import myloader15 as ls
    args.datapath = 'dataset/kitti2015_train/'

all_left_img, all_right_img, all_dwt_train, all_left_disp, test_left_img, test_right_img, test_dwt, test_disp = ls.dataloader(args.datapath)
TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(all_left_img, all_right_img, all_dwt_train,all_left_disp, True),
    batch_size=4, shuffle=True, num_workers=4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder(test_left_img, test_right_img, test_dwt, test_disp, False),
    batch_size=4, shuffle=False, num_workers=8, drop_last=False)

model = TANet(args.maxdisp)

if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))


# 学习率调整
def adjust_learning_rate(optimizer, epoch):
    warm_up = 0.02
    const_range = 0.6
    min_lr_rate = 0.05

    if epoch <= 500 * warm_up:
        lr = (1 - min_lr_rate) * 4.0e-4 / (500 * warm_up) * epoch + min_lr_rate * 4.0e-4
    elif 500 * warm_up < epoch <= 500 * const_range:
        lr = 4.0e-4
    else:
        lr = (min_lr_rate - 1) * 4.0e-4 / ((1 - const_range) * 500) * epoch + (1 - min_lr_rate * const_range) / (
                    1 - const_range) * 4.0e-4

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(imgL, imgR, dwt_img, disp_L):
    model.train()
    imgL     = Variable(torch.FloatTensor(imgL))
    imgR     = Variable(torch.FloatTensor(imgR))
    dwt_img  = Variable(torch.FloatTensor(dwt_img))  # 使用 dwt_img 代替 disp_pre
    disp_L   = Variable(torch.FloatTensor(disp_L))

    if args.cuda:
        imgL, imgR, dwt_img, disp_true = imgL.cuda(), imgR.cuda(), dwt_img.cuda(), disp_L.cuda()

    mask = (disp_true > 0)
    mask.detach_()

    optimizer.zero_grad()

    output = model(imgL, imgR, dwt_img)
    output = torch.squeeze(output, 1)
    loss = F.smooth_l1_loss(output[mask], disp_true[mask], size_average=True)

    loss.backward()
    optimizer.step()

    return loss.item()


def test(imgL, imgR, dwt_img, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    dwt_img = Variable(torch.FloatTensor(dwt_img))

    if args.cuda:
        imgL, imgR, dwt_img = imgL.cuda(), imgR.cuda(), dwt_img.cuda()

    start_time = time.time()
    with torch.no_grad():
        output = model(imgL, imgR, dwt_img)
    cost_time = time.time() - start_time

    pred_disp = output.data.cpu()
    pred_disp = pred_disp.squeeze(1)

    mask = (disp_true > 0)
    mask.detach_()
    epe = EPE_metric(pred_disp, disp_true, mask)
    D1 = D1_metric(pred_disp, disp_true, mask)
    Thres1 = Thres_metric(pred_disp, disp_true, mask, 1.0)
    Thres2 = Thres_metric(pred_disp, disp_true, mask, 2.0)
    Thres3 = Thres_metric(pred_disp, disp_true, mask, 3.0)
    err_pac = [epe, D1, Thres1, Thres2, Thres3]

    return err_pac, cost_time



def main():
    start_full_time = time.time()

    min_D1 = 100
    min_D1_epoch = 0
    writer = SummaryWriter(log_dir=args.logdir, flush_secs=5)

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        
        total_train_loss = 0
        total_times = 0
        total_epe = 0
        total_D1_t = 0
        total_T1 = 0
        total_T2 = 0
        total_T3 = 0
        adjust_learning_rate(optimizer, epoch)

       
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L,dwt_img) in tqdm(enumerate(TrainImgLoader)):
            loss = train(imgL_crop, imgR_crop, disp_crop_L,dwt_img)
            total_train_loss += loss
        # print("Shape of dwt_img in train:", dwt_img.shape)
        # print("Type of dwt_img in train:", type(dwt_img))
        train_loss = total_train_loss / len(TrainImgLoader)
        print(f'epoch {epoch} total training loss = {train_loss:.3f}')

        # 测试阶段，计算评估指标
        for batch_idx, (imgL, imgR, dwt_img, disp_L) in enumerate(TestImgLoader):
            err, cost_time = test(imgL, imgR, dwt_img, disp_L)
            total_times += cost_time / 4  # batch size = 4
            total_epe += err[0]
            total_D1_t += err[1]
            total_T1 += err[2]
            total_T2 += err[3]
            total_T3 += err[4]
        val_time = total_times / len(TestImgLoader)
        val_epe = total_epe / len(TestImgLoader)
        val_D1_t = total_D1_t / len(TestImgLoader) * 100
        val_T1 = total_T1 / len(TestImgLoader) * 100
        val_T2 = total_T2 / len(TestImgLoader) * 100
        val_T3 = total_T3 / len(TestImgLoader) * 100
        print(
            'average time = %.3f, average epe = %.3f, average D1_t = %.3f, average T1 = %.3f, average T2 = %.3f, average T3 = %.3f'
            % (val_time, val_epe, val_D1_t, val_T1, val_T2, val_T3))
      
        if val_D1_t < min_D1:
            min_D1 = val_D1_t
            min_D1_epoch = epoch
        if epoch > 280:
            savefilename = args.savemodel + f'finetune_{epoch}.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': train_loss,
                'test_loss': val_D1_t,
            }, savefilename)


        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('val_D1_t', val_D1_t, epoch)
        writer.add_scalar('val_epe', val_epe, epoch)
        writer.add_scalar('val_T1', val_T1, epoch)
        writer.add_scalar('val_T2', val_T2, epoch)
        writer.add_scalar('val_T3', val_T3, epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

    print(f'full finetune time = {(time.time() - start_full_time) / 3600:.2f} HR, '
          f'epoch {min_D1_epoch} get min_D1 = {min_D1:.3f}')

    writer.close()

if __name__ == '__main__':
    main()
