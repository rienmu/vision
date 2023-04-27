# coding=utf-8
import torchvision
from torch import nn


class ResNet(nn.Module):
    def __init__(self, num_classes=14):
        super(ResNet, self).__init__()
        BackBone = torchvision.models.__dict__['resnet50'](weights="ResNet50_Weights.DEFAULT")#
        add_block = []
        add_block += [nn.Linear(1000, 512)]
        add_block += [nn.ReLU(True)]
        add_block += [nn.Dropout(0.15)]
        add_block += [nn.Linear(512, num_classes)]
        add_block = nn.Sequential(*add_block)

        self.BackBone = BackBone
        self.add_block = add_block

    def forward(self, x):
        x = self.BackBone(x)
        x = self.add_block(x)

        return x



