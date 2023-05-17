#!/usr/bin/python3
#coding=utf-8
import os
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp
from utils import dataset_medical
from model.CHNet import CHNet
from visdom import Visdom
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter



def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)

def train(Dataset, Network):
    ## dataset
    # train_path = "/home/data/data/whb/PolypDataset/Train"
    train_path = "/home/data/data/whb/MIS_data/lung/Train"
    # cfg = Dataset.Config(datapath=train_path, savepath='./saved_model/CDNet_lung', batch=16, lr=0.05, momen=0.9, decay=5e-4, epoch=100)
    
    cfg = Dataset.Config(datapath=train_path, savepath='./saved_model/CHNet', mode='train', batch=32, lr=0.05, momen=0.9, decay=5e-4, epoch=100)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    ## network
    net = Network()
    net.train(True) 
    net.cuda()
   
    torch.backends.cudnn.enabled = False  # res2net does not support cudnn in py17

    ## parameter
    base, head = [], []


    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params':base}, {'params':head}], lr=cfg.lr, momentum=cfg.momen, weight_decay=cfg.decay, nesterov=True)
    global_step    = 0

    #实例化一个窗口
    # wind = Visdom()
    # #初始化窗口信息
    # wind.line(
    #             [0.],#Y 的第一个点坐标
    #             [0.],#X 的第一个点坐标
    #             win='train_loss',
    #             opts=dict(title='train_loss')
    # )

    writer = SummaryWriter(log_dir='logs')

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr
        optimizer.param_groups[1]['lr'] = (1-abs((epoch+1)/(cfg.epoch+1)*2-1))*cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            with amp.autocast(enabled=use_fp16):
                output = net(image)
    
                loss = structure_loss(output, mask)

            
            # print('mask:',mask.sum())
            writer.add_scalar(tag="mask.sum", # 可以暂时理解为图像的名字
                      scalar_value=mask.sum()/32,  # 纵坐标的值
                      global_step=global_step  # 当前是第几次迭代，可以理解为横坐标的值
                      )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if step %10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss=%.6f | mask=%.6f  '%(datetime.datetime.now(), global_step, epoch+1, cfg.epoch, optimizer.param_groups[1]['lr'],loss.item(),mask.sum()))
                # wind.line([loss],[step],win='train_loss',update='append')
                # time.sleep(0.5)
                
           
  

        if epoch>(cfg.epoch-5):
            torch.save(net.state_dict(), cfg.savepath+'/model-'+str(epoch+1))

if __name__=='__main__':
    torch.cuda.set_device(0)  # set your gpu device
    train(dataset_medical,CHNet)


