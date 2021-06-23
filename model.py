#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.rnn import GRU

CONFIG = {
    "loss_function": nn.MSELoss(),
    "hidden_size": 128,
    "num_layers": 1,
    "batch_first": False,
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
    _t: Tensor
    _t_prime: Tensor

    # NOTE: 我不曉得用其他類型的 AE 會有什麼差異，
    # 也許可以嘗試其他類型的，所以這部分就標記成NOTE。
    def __init__(
        self,
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

        # h function
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def forward(self, t):

        self._t = t
        self._t_prime = self.decoder(self._t)

        return self._t_prime


class ALComponent(nn.Module):

    y: Tensor
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
        batch_first: bool = CONFIG["batch_first"],
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
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )
            self.g = MODELS[mode](
                output_size,
                hidden_size,
                num_layers,
                batch_first=batch_first,
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
        self.ae = AutoEncoder(output_size, hidden_size)
        self.criterion = CONFIG["loss_function"]

    def forward(self, x, y):

        self.y = y
        self._s = self.f(x)
        self._s_prime = self.b(self._s)
        self._t = self.g(y)
        self._t_prime = self.ae(self._t)

        return self._s, self._t_prime

    def loss(self):

        return self.criterion(self._s_prime, self._t) + self.criterion(self._t_prime, self.y)


class Linear_AL(ALComponent):

    def __init__(self, *args, **kwargs) -> None:

        super(Linear_AL, self).__init__("Linear", *args, **kwargs)


class LSTM_AL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(self, *args, **kwargs) -> None:

        super(LSTM_AL, self).__init__("LSTM", *args, **kwargs)

    def forward(
        self,
        input: Tensor,
        output: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ):
        """https://github.com/pytorch/pytorch/blob/700df82881786f3560826f194aa9e04beeaa3fd8/torch/nn/modules/rnn.py#L659"""

        # TODO: determine the initial hidden state and the cell state of output.
        self.y = output
        self._s, (self._h_nx, c_nx) = self.f(input, hx)
        self._t, (self._h_ny, c_ny) = self.g(output, None)
        self._t_prime = self.ae(self._t)

        return self._s, self._t_prime

    def loss(self):

        # loss function for seq2seq model
        return self.criterion(self._h_nx, self._h_ny) + self.criterion(self._t_prime, self.y)


class GRU_AL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(self, *args, **kwargs) -> None:

        super(GRU_AL, self).__init__("GRU", *args, **kwargs)

    def forward(
        self,
        input: Tensor,
        output: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
    ):

        self.y = output
        self._s, self._h_nx = self.f(input, hx)
        self._t, self._h_ny = self.g(output, None)
        self._t_prime = self.ae(self._t)

        return self._s, self._t_prime

    def loss(self):

        # loss function for seq2seq model
        return self.criterion(self._h_nx, self._h_ny) + self.criterion(self._t_prime, self.y)


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

    inputs = torch.randn(8, 1, 4, requires_grad=True)
    outputs = torch.randn(8, 1, 4, requires_grad=True)
    model = LSTM_AL(4, 4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    epochs = 5
    for _ in range(epochs):

        x_out, y_out = model(inputs, outputs)
        loss = model.loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(inputs.shape)
        # print(outputs.shape)
        print(x_out.shape, y_out.shape)
        print(loss.item())


if __name__ == "__main__":
    test()
