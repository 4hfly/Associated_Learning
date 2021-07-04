#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import jsons
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

MODELS = {
    "MSE": nn.MSELoss,
    "Linear": nn.Linear,
    "LSTM": nn.LSTM,
    "LSTMCell": nn.LSTMCell,
    "GRU": nn.GRU,
    "GRUCell": nn.GRUCell
}

CONFIG = {
    "loss_function": "MSE",
    "hidden_size": 128,
    "num_layers": 1,
    "bias": True,
    "batch_first": False,
    "dropout": 0,
    "bidirectional": False,
}


class ALNN(nn.Module):

    def __init__(
        self,
        mode: str,
        input_size: int,
        output_size: int,
        hidden_size: int
    ):
        super(ALNN, self).__init__()

        self.layers = nn.Sequential(
            ALComponent(mode, input_size, output_size, hidden_size),
            # ALComponent(mode, hidden_size, hidden_size, 768)
        )

    def forward(self, x, y):

        return self.layers(x, y)


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
        bias: bool = CONFIG["bias"],
        batch_first: bool = CONFIG["batch_first"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
    ) -> None:

        super(ALComponent, self).__init__()
        # f function
        if mode == "Linear":
            self.f = MODELS[mode](input_size, input_size)
            self.g = MODELS[mode](output_size, hidden_size)

        elif mode == "LSTM" or mode == "GRU":
            self.f = MODELS[mode](
                input_size,
                hidden_size,
                num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )
            self.g = MODELS[mode](
                output_size,
                hidden_size,
                num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )

        # bridge function
        self.b = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        self.criterion = MODELS[CONFIG["loss_function"]]()

    def forward(self, x, y):

        self.y = y
        self._s = self.f(x)
        self._s_prime = self.b(self._s)
        if self.training:
            self._t = self.g(y)
            self._t_prime = self.decoder(self._t)
            return self._s.detach(), self._t.detach()
        else:
            self._t_prime = self.decoder(self._s_prime)
            return self._s.detach(), self._t_prime.detach()

    def loss(self):

        return self.criterion(self._s_prime, self._t) + self.criterion(self._t_prime, self.y)


class LinearAL(ALComponent):

    def __init__(self, *args, **kwargs) -> None:

        super(LinearAL, self).__init__("Linear", *args, **kwargs)


class LSTMAL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(self, *args, **kwargs) -> None:

        super(LSTMAL, self).__init__("LSTM", *args, **kwargs)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        hy: Optional[Tuple[Tensor, Tensor]] = None
    ):
        """https://github.com/pytorch/pytorch/blob/700df82881786f3560826f194aa9e04beeaa3fd8/torch/nn/modules/rnn.py#L659

        Args:
            x: (L, N, Hin) or (N, L, Hin)
            y: (L, N, Hout) or (N, L, Hout)
            hx: (D * num_layers, N, Hcell)
            hy: (D * num_layers, N, Hcell)

            L = sequence length
            N = batch size
            D = bidirectional

        Returns:

            x outputs: output x, (hx_n, cx_n)
            y outputs: output y, (hy_n, hy_n)

        """

        self.y = y
        self._s, (self._h_nx, c_nx) = self.f(x, hx)
        if self.training:
            self._t, (self._h_ny, c_ny) = self.g(y, hy)
            self._t_prime = self.decoder(self._t)
            return self._s.detach(), (self._h_nx, c_nx), self._t.detach(), (self._h_ny, c_ny)
        else:
            self._t_prime = self.decoder(self._s)
            return self._s.detach(), self._t_prime.detach()

    def loss(self):

        if self.training:
            # loss function for seq2seq model
            return self.criterion(self._h_nx, self._h_ny) + self.criterion(self._t_prime, self.y)
        else:
            self.criterion(self._t_prime, self.y)


class GRUAL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(self, *args, **kwargs) -> None:

        super(GRUAL, self).__init__("GRU", *args, **kwargs)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        hy: Optional[Tuple[Tensor, Tensor]] = None
    ):

        self.y = y
        self._s, self._h_nx = self.f(x, hx)
        self._t, self._h_ny = self.g(y, hy)
        if self.training:
            self._t_prime = self.decoder(self._t)
            return self._s.detach(), self._h_nx, self._t.detach(), self._h_ny
        else:
            self._t_prime = self.decoder(self._s)
            return self._s.detach(), self._t_prime.detach()

    def loss(self):

        if self.training:
            # loss function for seq2seq model
            return self.criterion(self._h_nx, self._h_ny) + self.criterion(self._t_prime, self.y)
        else:
            self.criterion(self._t_prime, self.y)


class ALComponentCell(nn.Module):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(
        self,
        mode: str,
        input_size: int,
        output_size: int,
        hidden_size: int,
        bias: bool = CONFIG["bias"]
    ) -> None:

        super(ALComponent, self).__init__()
        # f function
        self.f = MODELS[mode](
            input_size,
            hidden_size,
            bias=bias,
        )
        self.g = MODELS[mode](
            output_size,
            hidden_size,
            bias=bias,
        )

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        hy: Optional[Tuple[Tensor, Tensor]] = None
    ):
        self.y = y
        self._h_nx = self.f(x, hx)
        self._h_ny = self.g(y, hy)

        return self._h_nx, self._h_ny


class LSTMCellAL(ALComponentCell):

    _c_nx: Tensor
    _c_ny: Tensor

    def __init__(self, *args, **kwargs) -> None:
        super(LSTMCellAL, self).__init__("LSTMCell", *args, **kwargs)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        hy: Optional[Tuple[Tensor, Tensor]] = None
    ):
        self.y = y
        (self._h_nx, self._c_nx) = self.f(x, hx)
        (self._h_ny, self._c_ny) = self.g(y, hy)

        return (self._h_nx, self._c_nx), (self._h_ny, self._c_ny)


class GRUCellAL(ALComponentCell):

    def __init__(self, *args, **kwargs) -> None:
        super(GRUCellAL, self).__init__("LSTMCell", *args, **kwargs)


class ResNetAL(ALComponent):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py"""

    def __init__(self, *args, **kwargs) -> None:

        super(ResNetAL, self).__init__("ResNet", *args, **kwargs)


class VGGAL(ALComponent):
    """https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py"""

    def __init__(self, *args, **kwargs) -> None:

        super(VGGAL, self).__init__("VGG", *args, **kwargs)


class CNNAL(ALComponent):

    def __init__(self, *args, **kwargs) -> None:

        super(CNNAL, self).__init__("CNN", *args, **kwargs)


class MLPAL(ALComponent):

    def __init__(self, *args, **kwargs) -> None:

        super(MLPAL, self).__init__("MLP", *args, **kwargs)


def test():

    inputs = torch.randn(8, 1, 4)
    outputs = torch.randn(8, 1, 40)
    model = LSTMAL(4, 4, 2)
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


def load_parameters():

    global CONFIG
    with open("configs/hyperparams.json", "r", encoding="utf8") as f:
        CONFIG = json.load(f)


def save_parameters():

    with open("configs/hyperparameters.json", "w", encoding="utf8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, sort_keys=True, indent=3)


if __name__ == "__main__":
    test()
