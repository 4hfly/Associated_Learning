#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utils import LabelSmoothingLoss


MODELS = {
    "MSE": nn.MSELoss,
    "Linear": nn.Linear,
    "LSTM": nn.LSTM,
    "LSTMCell": nn.LSTMCell,
    "GRU": nn.GRU,
    "GRUCell": nn.GRUCell,
    "Emb":nn.Embedding
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
        hidden_size: Tuple[int, int],
        num_layers: int = CONFIG["num_layers"],
        bias: bool = CONFIG["bias"],
        batch_first: bool = CONFIG["batch_first"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
    ) -> None:

        super(ALComponent, self).__init__()
        # f function
        if mode == "Linear":
            self.f = MODELS[mode](input_size, hidden_size[0])
            self.g = MODELS[mode](output_size, hidden_size[1])

        elif mode == "LSTM" or mode == "GRU":
            self.f = MODELS[mode](
                input_size,
                hidden_size[0],
                num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )
            self.g = MODELS[mode](
                output_size,
                hidden_size[1],
                num_layers,
                bias=bias,
                batch_first=batch_first,
                dropout=dropout,
                bidirectional=bidirectional
            )

        # bridge function
        self.b = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Sigmoid()
        )

        # h function
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size[1], output_size),
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
            return self._s, self._t_prime

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
        self._s_prime = self.b(self._s)
        if self.training:
            self._t, (self._h_ny, c_ny) = self.g(y, hy)
            self._t_prime = self.decoder(self._t)
            # TODO: not sure which one is correct.
            # (self._h_nx, c_nx) -> (self._h_nx.detach(), c_nx.detach())
            return self._s.detach(), (self._h_nx, c_nx), self._t.detach(), (self._h_ny, c_ny)
        else:
            self._t_prime = self.decoder(self._s)
            return self._s, (self._h_nx, c_nx), self._t_prime

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
        hx: Optional[Tensor] = None,
        hy: Optional[Tensor] = None
    ):

        self.y = y
        self._s, self._h_nx = self.f(x, hx)
        self._s_prime = self.b(self._s)
        if self.training:
            self._t, self._h_ny = self.g(y, hy)
            self._t_prime = self.decoder(self._t)
            return self._s.detach(), self._h_nx, self._t.detach(), self._h_ny
        else:
            self._t_prime = self.decoder(self._s)
            return self._s, self._h_nx, self._t_prime

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


class EmbAL(nn.Module):
    def __init__(self, EmbAL, emb_dim, vocab_size=25000):
        self.f = nn.Embedding(vocab_size, emb_dim)
        self.g = nn.Embedding(vocab_size, emb_dim)
        self.decoder_f = nn.Linear(emb_dim, vocab_size)
        self.decoder_g = nn.Linear(emb_dim, vocab_size)
        self.labelsmoothingloss = LabelSmoothingLoss(0.1, 25000)

    def forward(self, x, y, reverse=False):
        '''
        input params:
            x: (src_len, batch_size)
            y: (tgt_len, batch_size)
        out params:
            emb_x: (src_len, batch_size, emb_dim)
            emb_y = (tgt_len, batch_size, emb_dim)
        '''

        emb_x = self.f(x)
        emb_y = self.g(y)
        loss_b = self.mse_loss(emb_x, emb_y, reverse) # bridge loss
        if not reverse:
            loss_d = self.decode_loss(emb_x, x, reverse)
        else:
            loss_d = self.decode_loss(emb_y, y, reverse)
        self.loss = loss_b + loss_d
        return emb_x, emb_y

    def loss(self):
        return self.loss

    def inference(self, x, tgt, reverse=False):

        '''
        input params:
            x: (tgt_len-1, batch_size, hidden)
            tgt: (tgt_len-1, batch_size)
        output params:
            out: (tgt_len-1, batch_size, vocab_size)
        '''

        if not reverse:
            out = F.logsoftmax(self.decoder_g(x))
        else:
            out = F.logsoftmax(self.decoder_f(x))
        return out

    def mse_loss(self, x, y):

        '''
        input params:
            x: (src_len, batch_size, emb_dim)
            y: (tgt_len, batch_size, emb_dim)
        '''
        x = x[x.nonzero(as_tuple=True)].view(x.size(0), x.size(1), -1).mean(1)
        y = y[y.nonzero(as_tuple=True)].view(y.size(0), y.size(1), -1).mean(1)
        self.loss_b = F.mse_loss(x, y)
    
    def decode_loss(self, pred, tgt, reverse=False):

        '''
        input params:
            pred: (tgt_len -1, batch_size, hidden)
            tgt: (tgt_len - 1, batch_size)
        parameters:
            word_prob: (tgt_len - 1, batch_size, vocab_size)

        '''

        if not reverse:
            word_prob = F.logsoftmax(self.decoder_g(pred)) # readout layer 
        else:
            word_prob = F.logsoftmax(self.decoder_f(pred)) # readout layer
        tgt_words_to_pred = torch.count_nonzero(tgt)
        prob = -self.labelsmoothingloss(word_prob.reshape(-1, word_prob.size(-1)), tgt[1:].view(-1)).view(-1, pred.size(1)).sum(0)
        prob = prob.sum() / pred.size(1)
        self.loss_d = prob


class GRUCellAL(ALComponentCell):

    def __init__(self, *args, **kwargs) -> None:
        super(GRUCellAL, self).__init__("LSTMCell", *args, **kwargs)


class ALNet(nn.Module):

    _mode = {
        "Linear": LinearAL,
        "LSTM": LSTMAL,
        "GRU": GRUAL,
    }

    def __init__(
        self,
        mode: str,
        input_size: int,
        output_size: int,
        hidden_size: List[Tuple[int, int]]
    ):
        super(ALNet, self).__init__()

        self.mode = mode
        model = self._mode[mode]
        # hidden_size: List[Tuple[int, int]]
        # e.g. [(256, 256), (128, 128), (64, 64)]
        # The following code will create an ordered dictionary, and the number
        # of layers depends on the length of list hidden_size. Each layer has
        # its own parameters like input_size or output_size. We may need this
        # data format for pipeline usage.
        # layers: {
        #     "layers_1": LSTMAL(300, 300, (256, 256)),
        #     "layers_2": LSTMAL(256, 256, (128, 128)),
        #     "layers_3": LSTMAL(128, 128, ( 64,  64))
        # }
        layers = [("layer_1", model(input_size, output_size, hidden_size[0]))]
        for i, h in enumerate(hidden_size[1:]):
            layers.append((
                f"layer_{i+2}",
                model(
                    input_size=hidden_size[i][0],
                    output_size=hidden_size[i][1],
                    hidden_size=h
                )
            ))

        self.layers = nn.ModuleDict(OrderedDict(layers))

    def forward(self, x, y, hx, hy, choice):

        if self.training:

            if self.mode == "Linear":
                s, t = self.layers[f"layer_{choice}"](x, y)
                return s, t

            elif self.mode == "LSTM" or self.mode == "Linear":
                s, hs, t, ht = self.layers[f"layer_{choice}"](x, y, hx, hy)
                return s, hs, t, ht

        else:

            if self.mode == "Linear":
                s, t_prime = self.layers[f"layer_{choice}"](x, y)
                return s, t_prime

            elif self.mode == "LSTM" or self.mode == "Linear":
                s, hs, t_prime = self.layers[f"layer_{choice}"](x, y, hx)
                return s, hs, t_prime


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