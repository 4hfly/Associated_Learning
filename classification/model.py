# -*- coding: utf-8 -*-
import json
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
# from torchtext.vocab import Vectors

CONFIG = {
    "hidden_size": (128, 128),
    "num_layers": 1,
    "bias": False,
    "batch_first": True,
    "dropout": 0.,
    "bidirectional": True,
    "vocab_size": (25000, 25000),
    "embedding_dim": (300, 128)
}


class CLS(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, emb=None):

        super(CLS, self).__init__()
        if emb:
            self.emb = emb
        else:
            self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            input_size=emb_dim, hidden_size=hid_dim, num_layers=2, dropout=0.2, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hid_dim*4, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.emb(x)
        output, (h, c) = self.lstm(x)
        h = h.reshape(h.size(1), -1)
        out = self.fc(h)
        return out


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

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.f = f
        self.g = g
        # birdge function
        self.bx = bx
        # self.by = by
        # decoder h function
        # self.dx = dx
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
            # self._s_prime = self.dx(self._s)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), self._t.detach()

        else:

            if not self.reverse:
                self._s = self.f(x)
                # self._t_prime = self.dy(self.bx(self._s))
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
        pretrained: int = None,
        padding_idx: int = 0,
        reverse: bool = False,
        lin: bool = False,
    ) -> None:

        if pretrained is not None:
            f = nn.Embedding.from_pretrained(
                pretrained, padding_idx=padding_idx) # freeze=False
        else:
            f = nn.Embedding(
                num_embeddings[0], embedding_dim[0], padding_idx=padding_idx)
        self.lin = lin
        print(self.lin)
        # TODO:
        if self.lin:
            g = nn.Sequential(
                nn.Linear(num_embeddings[1], embedding_dim[1], bias=False),
                nn.Tanh()
            )
        else:
            g = nn.Embedding(
                num_embeddings[1], embedding_dim[1], padding_idx=padding_idx)
        # bridge function
        bx = nn.Sequential(
            nn.Linear(embedding_dim[0], embedding_dim[1], bias=False),
            nn.Tanh()
        )

        by = None
        dx = None

        if num_embeddings[1] == 2:
            self.output_dim = 1
        else:
            self.output_dim = num_embeddings[1]

        dy = nn.Sequential(
            nn.Linear(embedding_dim[1], self.output_dim, bias=False),
            nn.Tanh()
        )
        # loss function
        cb = nn.MSELoss()
        ca = nn.MSELoss()

        super(EmbeddingAL, self).__init__(
            f, g, bx, by, dx, dy, cb, ca, reverse=reverse)

    def loss(self):

        p = self._s
        q = self._t

        p_nonzero = (p != 0.).sum(dim=1)
        p = p.sum(dim=1) / p_nonzero

        if not self.reverse:
            loss_b = self.criterion_br(self.bx(p), q)
            if self.output_dim == 1:
                loss_d = self.criterion_ae(
                    self._t_prime.squeeze(1), self.y.to(torch.float))
            else:
                loss_d = self.criterion_ae(
                    self._t_prime, self.y.to(torch.float))
        else:
            raise Exception()

        return loss_b + loss_d


class LinearAL(ALComponent):
    """
    For classification.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: Tuple[int, int],
        bias: bool = False,
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
        batch_first: bool = True,
        dropout: float = 0.,
        bidirectional: bool = False,
        reverse: bool = False
    ) -> None:

        if bidirectional:
            self.d = 2
        else:
            self.d = 1

        f = nn.LSTM(
            input_size,
            hidden_size[0],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        g = nn.Sequential(
            nn.Linear(output_size, hidden_size[1], bias=False),
            nn.Tanh()
        )
        # bridge function
        bx = nn.Sequential(
            nn.Linear(hidden_size[0] * self.d, hidden_size[1], bias=False),
            nn.Tanh()
        )
        by = None
        dx = None

        dy = nn.Sequential(
            nn.Linear(hidden_size[1], output_size, bias=False),
            nn.Tanh()
        )
        # loss function
        cb = nn.MSELoss(reduction='mean')
        ca = nn.MSELoss(reduction='mean')

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
        # print('lstm x', x.shape, 'y', y.shape)
        if self.training:

            self._s, (self._h_nx, c_nx) = self.f(x, hx)
            self._h_nx = self._h_nx.reshape(self._h_nx.size(1), -1)
            # print('hx', self._h_nx.shape)

            # self._s_prime = self.dx(self._h_nx)
            self._t = self.g(y)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), (self._h_nx.detach(), c_nx.detach()), self._t.detach()

        else:

            if not self.reverse:
                self._s, (self._h_nx, c_nx) = self.f(x, hx)
                self._h_nx = self._h_nx.view(
                    1, -1, self._h_nx.size(2) * self.d)
                self._t_prime = self.dy(self.bx(self._h_nx))
                return self._s.detach(), (self._h_nx.detach(), c_nx.detach()), self._t_prime.detach()
            else:
                raise Exception()

    def loss(self):

        # p = self._h_nx
        p = self._s[:, -1, :]
        # print('p', p.shape)
        q = self._t
        q = self._t
        # p = p.view(1, -1, p.size(2) * self.d)
        # print(p.shape)

        if not self.reverse:
            loss_b = self.criterion_br(self.bx(p), q)
            loss_d = self.criterion_ae(self._t_prime, self.y)
        else:
            raise Exception()

        return loss_b + loss_d


class TransformerEncoderAL(ALComponent):

    def __init__(
        self,
        d_model: Tuple[int, int],
        nhead: int,
        y_hidden: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True
    ) -> None:

        # TODO: pytorch v1.9.0 有 layer_norm_eps, batch_first 兩個參數，v1.8.1 沒有。
        f = nn.TransformerEncoderLayer(
            d_model[0], nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first)
        g = nn.Sequential(
            nn.Linear(d_model[1], y_hidden, bias=False),
            nn.Tanh()
        )
        bx = nn.Sequential(
            nn.Linear(d_model[0], y_hidden, bias=False),
            nn.Tanh()
        )
        by = None
        dx = None
        dy = nn.Sequential(
            nn.Linear(y_hidden, d_model[1], bias=False),
            nn.Tanh()
        )
        cb = nn.MSELoss(reduction='mean')
        ca = nn.MSELoss(reduction='mean')

        super().__init__(f, g, bx, by, dx, dy, cb, ca, reverse=False)

    def forward(self, x, y):

        # TODO: 還需要 src mask 的參數設定。
        return super().forward(x, y)

    def loss(self):

        # NOTE: v1.8.1 預設 batch_first = False。
        p = self._s[:, -1, :]
        q = self._t

        if not self.reverse:
            loss_b = self.criterion_br(self.bx(p), q)
            loss_d = self.criterion_ae(self._t_prime, self.y)
        else:
            raise Exception()

        return loss_b + loss_d


def load_parameters():

    global CONFIG
    with open("configs/hyperparams.json", "r", encoding="utf8") as f:
        CONFIG = json.load(f)


def save_parameters():

    with open("configs/hyperparameters.json", "w", encoding="utf8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, sort_keys=True, indent=3)


if __name__ == "__main__":
    save_parameters()
