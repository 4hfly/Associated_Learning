#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.modules.module import Module


CONFIG = {
    "loss_function": nn.MSELoss(),
    "hidden_size": 128,
    "num_layers": 1,
    "batch_size": 1,
    "dropout": 0,
    "bidirectional": False,
}

MODELS = {
    "Linear": nn.Linear,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
}


class AutoEncoder(nn.Module):

    #       _t: g(y) = encoded y
    # _t_prime: h(g(y)) = ae output
    y: Tensor
    _t: Tensor
    _t_prime: Tensor

    # NOTE: 我不曉得用其他類型的 AE 會有什麼差異，
    # 也許可以嘗試其他類型的，所以這部分就標記成NOTE。
    def __init__(
        self,
        g: Module,
        input_size: int,
        hidden_size: int
    ) -> None:
        r"""

        Args:
            mode: for g function
            input_size: y_size
            hidden_size:
        """
        super(AutoEncoder, self).__init__()

        # g function
        self.encoder = g
        # h function
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )
        self.criterion = CONFIG["loss_function"]

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

    def __init__(
        self,
        mode: str,
        input_size: int,
        output_size: int,
        hidden_size: int,
        num_layers: int = CONFIG["num_layers"],
        batch_size: int = CONFIG["batch_size"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
    ) -> None:

        super(ALComponent, self).__init__()
        # f function
        # TODO: 這麼寫主要是為了擴充性，假如各種網路拆開來寫不使用繼承，
        # 那麼 f_function 就不需要這麼多 if else。
        if mode == "Linear":
            self.f = MODELS[mode](input_size, input_size)
            self.g = MODELS[mode](output_size, hidden_size)

        # NOTE: Pytorch 自己會把 dropout 轉成 float。
        elif mode == "LSTM" or mode == "GRU":
            self.f = MODELS[mode](
                input_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )
            self.g = MODELS[mode](
                output_size,
                hidden_size,
                num_layers,
                dropout=dropout,
                bidirectional=bidirectional
            )

        # bridge function
        # TODO: 不確定這個 bridge 該如何設計。
        self.b = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )
        self.batch_size = batch_size
        self.ae = AutoEncoder(self.g, output_size, hidden_size)
        self.criterion = CONFIG["loss_function"]

    def forward(self, x, y):

        self._s = self.f(x)
        self._s_prime = self.b(self._s)
        self._t, self._t_prime = self.ae(y)

        return self._s, self._t_prime

    def loss(self):

        return self.criterion(self._s_prime, self._t) + self.ae.loss()


class Linear_AL(ALComponent):

    def __init__(self, *args, **kwargs) -> None:

        super(Linear_AL, self).__init__("Linear", *args, **kwargs)


class LSTM_AL(ALComponent):

    def __init__(self, *args, **kwargs) -> None:

        super(LSTM_AL, self).__init__("LSTM", *args, **kwargs)

    def forward(self, input, hx=None):

        return


class ResNet_AL(ALComponent):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"""

    def __init__(self, *args, **kwargs) -> None:

        super(ResNet_AL, self).__init__("ResNet_AL", *args, **kwargs)


class VGG_AL(ALComponent):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""

    def __init__(self, *args, **kwargs) -> None:

        super(VGG_AL, self).__init__("VGG_AL", *args, **kwargs)


class CNN_AL(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        super(CNN_AL, self).__init__("CNN_AL", *args, **kwargs)


class MLP_AL(nn.Module):

    def __init__(self, *args, **kwargs) -> None:

        super(MLP_AL, self).__init__("MLP_AL", *args, **kwargs)


def test():

    inputs = torch.randn(8)
    outputs = torch.randn(8)
    model = Linear_AL(len(inputs), len(outputs), 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for _ in range(epochs):

        x_out, y_out = model(inputs, outputs)
        loss = model.loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(inputs)
        print(outputs)
        print(x_out, y_out)
        print(loss.item())


if __name__ == "__main__":
    test()
