#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from torch import Tensor


class AutoEncoder(nn.Module):
    r"""

    """

    #    _t   : g(y) = encoded y
    # _t_prime: h(g(y)) = ae output
    y: Tensor
    _t: Tensor
    _t_prime: Tensor

    # NOTE: 論文上的版本應該是 linear，不過我不曉得用其他類型的 AE 會有什麼差異，
    # 也許可以嘗試其他類型的，所以這部分就標記成NOTE。
    def __init__(self, input_size, hidden_size=2) -> None:
        """
        TODO: 看起來只有一層，loss func. 是 MSE，optimizer 是 Adam，放著備忘用。

        Args:
            input_size: the total pixels of an image
            hidden_size:
        """
        super(AutoEncoder, self).__init__()

        # g function
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        # h function
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.criterion = nn.MSELoss()

    def forward(self, y):

        self.y = y
        self._t = self.encoder(self.y)
        self._t_prime = self.decoder(self._t)

        return self._t, self._t_prime

    def loss(self):
        """should never call this function."""

        return self.criterion(self._t_prime, self.y)


class ALComponent(nn.Module):

    _s: Tensor
    _t: Tensor
    _s_prime: Tensor
    _t_prime: Tensor

    def __init__(self, x_size, y_size, hidden_size=2) -> None:

        super(ALComponent, self).__init__()
        # ELU:
        # https://pytorch.org/docs/stable/generated/torch.nn.ELU.html#torch.nn.ELU
        self.f = nn.ELU()

        # bridge function
        # TODO: 不確定這個 bridge 該如何設計，看原版用 sigmoid，我先用 linear 確保 size
        # 可調整。
        self.b = nn.Sequential(
            nn.Linear(x_size, hidden_size),
            nn.Sigmoid()
        )
        self.ae = AutoEncoder(y_size)
        self.criterion = nn.MSELoss()

    def forward(self, x, y):

        self._s = self.f(x)
        self._s_prime = self.b(self._s)
        self._t, self._t_prime = self.ae(y)

        return self._s, self._t_prime

    def loss(self):

        return self.criterion(self._s_prime, self._t) + self.ae.loss()


class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.x = torch.ones(8)
        self.y = torch.zeros(8)
        self.model = ALComponent(8, 8)

    def forward(self):

        x_out, y_out = self.model(self.x, self.y)
        return x_out, y_out

    def loss(self):

        return self.model.loss()


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

    def forward(self, x):

        return self.layers(x)


class MLP(nn.Module):
    """"""

    # TODO: 還有更改空間，原版似乎複雜一點。
    def __init__(self, input_size, output_size, hidden_size=1024) -> None:
        """"""
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):

        x = self.flatten(x)
        return self.layers(x)


def test():

    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for _ in range(3):

        x_out, y_out = model()
        loss = model.loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(x_out, y_out)
        print(loss.item())


if __name__ == "__main__":
    test()
