#!/usr/bin/python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2net50_backbone import Backbone_Res2Net50_in3


class CHNet(nn.Module):
    def __init__(self):
        super(CHNet, self).__init__()

  
        self.encode1, self.encode2, self.encode3, self.encode4, _ = Backbone_Res2Net50_in3()

        self.classifier = nn.Conv2d(1024, 1, kernel_size=1, stride=1)
    
     #    self.initialize()

    def forward(self, x, shape=None):
        in_data_1 = self.encode1(x)  # [4,64,192,192]
        in_data_2 = self.encode2(in_data_1)  # [4,256,96,96]
        in_data_3 = self.encode3(in_data_2)  # [4,512,48,48]
        in_data_4 = self.encode4(in_data_3)  # [4,1024,24,24]       
        # in_data_5 = self.encode5(in_data_4)  # [4,2048,12,12]

        shape = x.size()[2:]

        out_data = self.classifier(in_data_4)
        out_data = F.interpolate(out_data, size=shape, mode='bilinear')
        
        
        return out_data

if __name__ == "__main__":
    model = CHNet()
    input = torch.autograd.Variable(torch.randn(4, 3, 352, 352))
    output = model(input)

    print("pre:",output.shape)

   