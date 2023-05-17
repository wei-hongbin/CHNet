#!/usr/bin/python3
# coding=utf-8
from thop import profile, clever_format
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2net50_backbone import Backbone_Res2Net50_in3


class CHNet(nn.Module):
    def __init__(self):
        super(CHNet, self).__init__()

        self.encode1, self.encode2, self.encode3, self.encode4, _ = Backbone_Res2Net50_in3()    
        self.tf1 = nn.Conv2d(64,64,1)
        self.tf2 = nn.Conv2d(256,64,1)
        self.tf3 = nn.Conv2d(512,64,1)
        self.tf4 = nn.Conv2d(1024,64,1)  

        self.classifier = nn.Sequential(
                nn.Conv2d(256, 32, 1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
                nn.Conv2d(32, 1, 1)
              )
 

    def forward(self, x, shape=None):
        in_data_1 = self.encode1(x)  # [4,64,192,192]
        in_data_2 = self.encode2(in_data_1)  # [4,256,96,96]
        in_data_3 = self.encode3(in_data_2)  # [4,512,48,48]
        in_data_4 = self.encode4(in_data_3)  # [4,1024,24,24]       
     
        tf1 = self.tf1(in_data_1)
        tf2 = self.tf2(in_data_2)
        tf3 = self.tf3(in_data_3)
        tf4 = self.tf4(in_data_4)
        
        shape = x.size()[2:]

    
        fea1 = F.interpolate(tf1, size=shape, mode='bilinear')
        fea2 = F.interpolate(tf2, size=shape, mode='bilinear')
        fea3 = F.interpolate(tf3, size=shape, mode='bilinear')
        fea4 = F.interpolate(tf4, size=shape, mode='bilinear')

        fea = torch.cat((fea1,fea2,fea3,fea4),dim=1)


        

        out_data = self.classifier(fea)

        return out_data

if __name__ == "__main__":
    model = CHNet().cuda()
    input = torch.autograd.Variable(torch.randn(1, 3, 256, 256)).cuda()
    # output = model(input)

    # print("pre:",output.shape)
    flops,params = profile(model,inputs=(input,))
    flops,params = clever_format([flops,params],"%.3f")

    print('flops:'+str(flops))
    print('params:'+str(params))