#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU

class AutoEncoder(nn.Module):
    """"""

    # TODO: 論文上的版本應該是 linear，不過我不曉得用其他類型的 AE 會有什麼差異，
    # 所以這部分就標記成TODO。
    def __init__(self, input_size, hidden_size=64) -> None:
        """
        TODO: 看起來只有一層，loss func 是 MSE，optimizer 是 Adam，放著備忘用。

        Args:
            input_size: the total pixels of an image
            hidden_size:
        """
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def forward(self, y):

        y = self.encoder(y)
        y = self.decoder(y)

        return y


class ResNet(nn.Module):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"""

    def __init__(self) -> None:
        """"""
        super(ResNet, self).__init__()


class VGG(nn.Module):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""

    def __init__(self) -> None:
        """"""
        super(VGG, self).__init__()


class CNN(nn.Module):

    def __init__(self, input_size, output_size) -> None:
        """"""
        super(CNN, self).__init__()
        self.kernel_size = 3
        self.stride = 2
        self.layers = nn.Sequential(
            nn.conv2d(input_size, output_size, self.kernel_size, self.stride),
            nn.ReLU(),
            nn.MaxPool2d(),
        )

class MLP(nn.Module):
    """"""

    # TODO: 看著 code 改的，別人的 MLP 不是長這樣
    def __init__(self, input_size, output_size, hidden_size=1024) -> None:
        """"""
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(), # TODO: 不確定是不是我誤會了啥
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size * 5),
            nn.ReLU(),
            nn.Linear(hidden_size * 5, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):

        return self.layers(x)