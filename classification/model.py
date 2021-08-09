# -*- coding: utf-8 -*-
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torchtext.vocab import Vectors

CONFIG = {
    "loss_function": "MSE",
    "decoder": "attn",
    "hidden_size": 128,
    "num_layers": 1,
    "bias": True,
    "batch_first": False,
    "dropout": 0.,
    "bidirectional": True,
    "vocab_size": (25000, 25000),
    "embedding_dim": (300, 128)
}


class ALComponent(nn.Module):

    x: Tensor
    y: Tensor
    _s: Tensor
    _t: Tensor
    _s_prime: Tensor
    _t_prime: Tensor

    def __init__(
        self,
        f: nn.Module,
        g: nn.Module,
        bx: nn.Module,
        by: nn.Module,
        dx: nn.Module,
        dy: nn.Module,
        cb: nn.Module,
        ca: nn.Module,
        reverse: bool = False
    ) -> None:

        super(ALComponent, self).__init__()

        self.f = f
        self.g = g
        # birdge function
        self.bx = bx
        self.by = by
        # decoder h function
        self.dx = dx
        self.dy = dy
        # loss function for bridge and auto-encoder
        self.criterion_br = cb
        self.criterion_ae = ca

        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.reverse = reverse

    def forward(self, x, y):

        self.x = x
        self.y = y

        if self.training:

            self._s = self.f(x)
            self._s_prime = self.dx(self._s)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), self._t.detach()

        else:

            if not self.reverse:
                self._s = self.f(x)
                self._t_prime = self.dy(self.bx(self._s))
                return self._s.detach(), self._t_prime.detach()
            else:
                self._t = self.g(x)
                self._s_prime = self.dx(self.by(self._t))
                return self._t.detach(), self._s_prime.detach()

    def loss(self):

        if not self.reverse:
            loss_b = self.criterion_br(self.bx(self._s), self._t)
            loss_d = self.criterion_ae(self._t_prime, self.y)
        else:
            loss_b = self.criterion_br(self.by(self._t), self._s)
            loss_d = self.criterion_ae(self._s_prime, self.x)

        return loss_b + loss_d


class EmbeddingAL(ALComponent):
    """
    For classification.
    """

    def __init__(
        self,
        num_embeddings: Tuple[int, int],
        embedding_dim: Tuple[int, int],
        pretrained: Vectors = None,
        padding_idx: int = 0,
        reverse: bool = False
    ) -> None:

        if pretrained:
            embeddings = torch.FloatTensor(pretrained.vectors)
            f = nn.Embedding.from_pretrained(
                embeddings, padding_idx=padding_idx)
        else:
            f = nn.Embedding(
                num_embeddings[0], embedding_dim[0], padding_idx=padding_idx)
        g = nn.Embedding(
            num_embeddings[1], embedding_dim[1], padding_idx=padding_idx)
        # bridge function
        bx = nn.Sequential(
            nn.Linear(embedding_dim[0], embedding_dim[1]),
            nn.Sigmoid()
        )
        by = nn.Sequential(
            nn.Linear(embedding_dim[1], embedding_dim[0]),
            nn.Sigmoid()
        )
        # h function
        dx = nn.Linear(embedding_dim[0], num_embeddings[0])
        dy = nn.Linear(embedding_dim[1], num_embeddings[1])
        # loss function
        cb = nn.MSELoss()
        ca = nn.BCEWithLogitsLoss()

        super(EmbeddingAL, self).__init__(
            f, g, bx, by, dx, dy, cb, ca, reverse=reverse)


class LinearAL(ALComponent):
    """
    For classification.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: Tuple[int, int],
        bias: bool = True,
        reverse: bool = False
    ) -> None:

        f = nn.Linear(in_features, hidden_size[0], bias=bias)
        g = nn.Linear(out_features, hidden_size[1], bias=bias)
        # bridge function
        bx = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Sigmoid()
        )
        by = nn.Sequential(
            nn.Linear(hidden_size[1], hidden_size[1]),
            nn.Sigmoid()
        )
        # h function
        dx = nn.Sequential(
            nn.Linear(hidden_size[0], in_features),
            nn.Sigmoid()
        )
        dy = nn.Sequential(
            nn.Linear(hidden_size[1], out_features),
            nn.Sigmoid()
        )
        # loss function
        cb = nn.MSELoss()
        ca = nn.MSELoss()

        super(LinearAL, self).__init__(
            f, g, bx, by, dx, dy, cb, ca, reverse=reverse)


class LSTMAL(ALComponent):
    """
    For classification.
    """

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Tuple[int, int],
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.,
        bidirectional: bool = False,
        reverse: bool = False
    ) -> None:

        f = nn.LSTM(
            input_size,
            hidden_size[0],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        g = nn.Linear(output_size, hidden_size[1])
        # bridge function
        bx = nn.Sequential(
            nn.Linear(hidden_size[0], hidden_size[1]),
            nn.Sigmoid()
        )
        by = nn.Sequential(
            nn.Linear(hidden_size[1], hidden_size[0]),
            nn.Sigmoid()
        )
        # h function
        dx = nn.LSTM(hidden_size[0], input_size)
        dy = nn.Sequential(
            nn.Linear(hidden_size[1], output_size),
            nn.Sigmoid()
        )
        # loss function
        cb = nn.MSELoss()
        ca = nn.MSELoss()

        super(LSTMAL, self).__init__(
            f, g, bx, by, dx, dy, cb, ca, reverse=reverse)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None,
        hy: Optional[Tuple[Tensor, Tensor]] = None
    ):
        """https://github.com/pytorch/pytorch/blob/700df82881786f3560826f194aa9e04beeaa3fd8/torch/nn/modules/rnn.py#L659\n
        Args:\n
            x: (L, N, Hin) or (N, L, Hin)\n
            y: (L, N, Hout) or (N, L, Hout)\n
            hx: ((D * num_layers, N, Hout), (D * num_layers, N, Hcell))\n
            hy: ((D * num_layers, N, Hout), (D * num_layers, N, Hcell))\n
            L = sequence length\n
            N = batch size\n
            D = bidirectional\n
        Returns:\n
            x outputs: output x, (hx_n, cx_n)\n
            y outputs: output y, (hy_n, hy_n)\n
        """

        self.x = x
        self.y = y

        if self.training:

            self._s, (self._h_nx, c_nx) = self.f(x, hx)
            self._s_prime = self.dx(self._s)
            self._t = self.g(y, hy)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), (self._h_nx, c_nx), self._t.detach()

        else:

            if not self.reverse:
                self._s, (self._h_nx, c_nx) = self.f(x, hx)
                self._t_prime = self.dy(self.bx(self._s))
                return self._s.detach(), (self._h_nx, c_nx), self._t_prime.detach()
            else:
                raise Exception()

    def loss(self):

        if not self.reverse:
            loss_b = self.criterion_br(self.bx(self._s), self._t)
            loss_d = self.criterion_ae(self._t_prime, self.y)
        else:
            raise Exception()

        return loss_b + loss_d
