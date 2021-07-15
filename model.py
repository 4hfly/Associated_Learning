#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from utils import LabelSmoothingLoss

MODELS = {
    "MSE": nn.MSELoss,
    "Embedding": nn.Embedding,
    "Linear": nn.Linear,
    "LSTM": nn.LSTM,
    "LSTMCell": nn.LSTMCell,
    "GRU": nn.GRU,
    "GRUCell": nn.GRUCell,
    "attention": None
}

CONFIG = {
    "loss_function": "MSE",
    "decoder": "attn",
    "hidden_size": 512,
    "num_layers": 1,
    "bias": True,
    "batch_first": False,
    "dropout": 0,
    "bidirectional": False,
    "vocab_size": (25000, 25000),
    "embedding_dim": (256, 256)
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
        mode: str,
        f: nn.Module,
        g: nn.Module,
        bx: nn.Module,
        by: nn.Module,
        dx: nn.Module,
        dy: nn.Module,
        reverse: bool = False
    ) -> None:

        super(ALComponent, self).__init__()

        self.f = f
        self.g = g
        # birdge function
        self.bx = bx
        self.by = by
        # h function
        self.dx = dx
        self.dy = dy
        self.dropout = nn.Dropout(CONFIG["dropout"])
        self.criterion = MODELS[CONFIG["loss_function"]]()
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

        if self.reverse:
            loss_b = self.criterion(self.bx(self._s), self._t)
            loss_d = self.criterion(self._t_prime, self.y)
        else:
            loss_b = self.criterion(self.by(self._t), self._s)
            loss_d = self.criterion(self._s_prime, self.x)

        return loss_b + loss_d

    def get_attention_mask(self, src_encodings: Tensor, src_sents_len: List[int]) -> Tensor:

        src_sent_masks = torch.zeros(
            src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to(src_encodings.device)

    def decode(self, src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var, tgt_emb):

        # (batch_size, src_sent_len, hidden_size)
        if not self.reverse:
            src_encoding_att_linear = self.att_src_g_linear(src_encodings)
        else:
            src_encoding_att_linear = self.att_src_f_linear(src_encodings)

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hid_dim,
                              device=src_encodings.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding

        tgt_word_embeds = tgt_emb(tgt_sents_var)  # tgt_emb is a layer

        h_tm1 = decoder_init_vec

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)

                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, src_encodings,
                                                      src_encoding_att_linear, src_sent_masks, self.reverse)

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    def step(self, x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks):

        # h_t: (batch_size, hidden_size)
        if not self.reverse:
            h_t, cell_t = self.decoder_g(x, h_tm1)
        else:
            h_t, cell_t = self.decoder_f(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(
            h_t, src_encodings, src_encoding_att_linear, src_sent_masks)

        if not self.reverse:
            att_t = torch.tanh(
                self.att_vec_g_linear(torch.cat([h_t, ctx_t], 1)))
        else:
            att_t = torch.tanh(
                self.att_vec_f_linear(torch.cat([h_t, ctx_t], 1)))

        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask):

        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear,
                               h_t.unsqueeze(2)).squeeze(2)

        if mask is not None:
            att_weight.data.masked_fill_(mask.bool(), -float("inf"))

        m = nn.Softmax(dim=-1)
        softmaxed_att_weight = m(att_weight)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(
            softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight


class LinearAL(ALComponent):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_size: Tuple[int, int],
        bias: bool = True,
        reverse: bool = False
    ) -> None:

        mode = "Linear"
        f = MODELS[mode](in_features, hidden_size[0], bias=bias)
        g = MODELS[mode](out_features, hidden_size[1], bias=bias)
        # bridge function
        bx = nn.Linear(hidden_size[0], hidden_size[1])
        by = nn.Linear(hidden_size[1], hidden_size[0])
        # h function
        dx = nn.Linear(hidden_size[0], in_features)
        dy = nn.Linear(hidden_size[1], out_features)

        super(LinearAL, self).__init__(
            mode, f, g, bx, by, dx, dy, reverse=reverse)


class LSTMAL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Tuple[int, int],
        num_layers: int = CONFIG["num_layers"],
        bias: bool = CONFIG["bias"],
        batch_first: bool = CONFIG["batch_first"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
        reverse: bool = False
    ) -> None:

        mode = "LSTM"
        f = MODELS[mode](
            input_size,
            hidden_size[0],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        g = MODELS[mode](
            output_size,
            hidden_size[1],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        # bridge function
        bx = nn.Linear(hidden_size[0], hidden_size[1])
        by = nn.Linear(hidden_size[1], hidden_size[0])
        # h function
        dx = nn.Linear(hidden_size[0], input_size)
        dy = nn.Linear(hidden_size[1], output_size)

        super(LSTMAL, self).__init__(
            mode, f, g, bx, by, dx, dy, reverse=reverse)

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
            hx: ((D * num_layers, N, Hout), (D * num_layers, N, Hcell))
            hy: ((D * num_layers, N, Hout), (D * num_layers, N, Hcell))

            L = sequence length
            N = batch size
            D = bidirectional

        Returns:

            x outputs: output x, (hx_n, cx_n)
            y outputs: output y, (hy_n, hy_n)

        """

        self.x = x
        self.y = y

        if self.training:

            self._s, (self._h_nx, c_nx) = self.f(x, hx)
            self._s_prime = self.dx(self._s)
            self._t, (self._h_ny, c_ny) = self.g(y, hy)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), (self._h_nx, c_nx), self._t.detach(), (self._h_ny, c_ny)

        else:

            if not self.reverse:
                self._s, (self._h_nx, c_nx) = self.f(x, hx)
                self._t_prime = self.dy(self.bx(self._s))
                return self._s.detach(), (self._h_nx, c_nx), self._t_prime.detach()
            else:
                self._t, (self._h_ny, c_ny) = self.g(y, hy)
                self._s_prime = self.dx(self.by(self._t))
                return self._t.detach(), (self._h_ny, c_ny), self._s_prime.detach()

    def loss(self):

        if not self.reverse:
            input = self.bx(self._h_nx).view(-1, self._h_nx.size()[2])
            target = self._h_ny.view(-1, self._h_ny.size()[2])
            loss_b = self.criterion(input, target)
            # loss_d = self.criterion(self._t_prime, self.y)
            return loss_b
        else:
            input = self.by(self._h_ny).view(-1, self._h_ny.size()[2])
            target = self._h_nx.view(-1, self._h_nx.size()[2])
            loss_b = self.criterion(input, target)
            # loss_d = self.criterion(self._s_prime, self.x)

            return loss_b


class GRUAL(ALComponent):

    _h_nx: Tensor
    _h_ny: Tensor

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: Tuple[int, int],
        num_layers: int = CONFIG["num_layers"],
        bias: bool = CONFIG["bias"],
        batch_first: bool = CONFIG["batch_first"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
        reverse: bool = False
    ) -> None:

        mode = "GRU"
        f = MODELS[mode](
            input_size,
            hidden_size[0],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        g = MODELS[mode](
            output_size,
            hidden_size[1],
            num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional
        )
        # bridge function
        bx = nn.Linear(hidden_size[0], hidden_size[1])
        by = nn.Linear(hidden_size[1], hidden_size[0])
        # h function
        dx = nn.Linear(hidden_size[0], input_size)
        dy = nn.Linear(hidden_size[1], output_size)

        super(LSTMAL, self).__init__(
            mode, f, g, bx, by, dx, dy, reverse=reverse)

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        hx: Optional[Tensor] = None,
        hy: Optional[Tensor] = None
    ):

        self.x = x
        self.y = y

        if self.training:

            self._s, self._h_nx = self.f(x, hx)
            self._s_prime = self.dx(self._s)
            self._t, self._h_ny = self.g(y, hy)
            self._t_prime = self.dy(self._t)
            return self._s.detach(), self._h_nx, self._t.detach(), self._h_ny

        else:

            if not self.reverse:
                self._s, self._h_nx = self.f(x, hx)
                self._t_prime = self.dy(self.bx(self._s))
                return self._s.detach(), self._h_nx, self._t_prime.detach()
            else:
                self._t, self._h_ny = self.g(y, hy)
                self._s_prime = self.dx(self.by(self._t))
                return self._t.detach(), self._h_ny, self._s_prime.detach()

    def loss(self):

        # TODO: not readable
        if not self.reverse:
            loss_b = self.criterion(self.bx(
                self._h_nx), self._h_ny).view(-1, self._h_ny.size()[1], self._h_ny.size()[2])
            loss_d = self.criterion(
                self._t_prime, self.y).view(-1, self.y.size()[1], self.y.size()[2])
            return loss_b + loss_d
        else:
            loss_b = self.criterion(self.self._h_nx, self.by(
                self._h_ny)).view(-1, self._h_nx.size()[1], self._h_nx.size()[2])
            loss_d = self.criterion(
                self._s_prime, self.x).view(-1, self.x.size()[1], self.x.size()[2])
            return loss_b + loss_d


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


class EmbeddingAL(ALComponent):

    def __init__(
        self,
        num_embeddings: Tuple[int, int],
        embedding_dim: Tuple[int, int],
        padding_idx: int = 0,
        reverse: bool = False
    ) -> None:

        mode = "Embedding"
        f = MODELS[mode](num_embeddings[0], embedding_dim[0],
                         padding_idx=padding_idx)
        g = MODELS[mode](num_embeddings[1], embedding_dim[1],
                         padding_idx=padding_idx)
        # bridge function
        bx = nn.Linear(embedding_dim[0], embedding_dim[1])
        by = nn.Linear(embedding_dim[1], embedding_dim[0])
        # h function
        dx = nn.Linear(embedding_dim[0], num_embeddings[0])
        dy = nn.Linear(embedding_dim[1], num_embeddings[1])

        super(EmbeddingAL, self).__init__(
            mode, f, g, bx, by, dx, dy, reverse=reverse)

        self.labelsmoothingloss = LabelSmoothingLoss(
            0.1, num_embeddings[int(reverse)])

    def loss(self):

        s = self._s
        t = self._t
        s = s[s.nonzero(as_tuple=True)].view(-1, s.size(1), s.size(2)).mean(0)
        t = t[t.nonzero(as_tuple=True)].view(-1, t.size(1), t.size(2)).mean(0)

        if not self.reverse:
            loss_b = self.criterion(s, t)
            loss_d = self.decode_loss(self._t, self.y)
        else:
            loss_b = self.criterion(t, s)
            loss_d = self.decode_loss(self._s, self.x)

        return loss_b + loss_d

    def decode_loss(self, pred, label):

        m = nn.LogSoftmax(-1)
        if not self.reverse:
            word_prob = m(self.dx(pred))
        else:
            word_prob = m(self.dy(pred))
        # TODO: 這個變數沒用到？
        tgt_words_to_pred = torch.count_nonzero(label)
        prob = -self.labelsmoothingloss(word_prob.reshape(-1, word_prob.size(-1)),
                                        label.view(-1)).view(-1, pred.size(1)).sum(0)
        prob = prob.sum() / pred.size(1)

        return prob


class ALNet(nn.Module):

    _mode = {
        "Linear": LinearAL,
        "LSTM": LSTMAL,
        "GRU": GRUAL,
    }

    def __init__(
        self,
        mode: str,
        hidden_size: List[Tuple[int, int]],
        num_embeddings: Tuple[int, int] = CONFIG["vocab_size"],
        embedding_dim: Tuple[int, int] = CONFIG["embedding_dim"],
        padding_idx: int = 0,
        num_layers: int = CONFIG["num_layers"],
        bias: bool = CONFIG["bias"],
        batch_first: bool = CONFIG["batch_first"],
        dropout: float = CONFIG["dropout"],
        bidirectional: bool = CONFIG["bidirectional"],
        reverse: bool = False
    ) -> None:
        """
        Args:
            mode: ["Linear", "LSTM", "GRU"]
        """
        super(ALNet, self).__init__()

        self.mode = mode
        model = self._mode[mode]
        self.emb = EmbeddingAL(
            num_embeddings, embedding_dim, padding_idx=padding_idx, reverse=reverse)
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
        kwargs = {
            "num_layers": num_layers,
            "bias": bias,
            "batch_first": batch_first,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        layers = [
            ("layer_1", model(embedding_dim[0], embedding_dim[1], hidden_size[0], **kwargs))]
        for i, h in enumerate(hidden_size[1:]):
            layers.append((
                f"layer_{i+2}",
                model(
                    input_size=hidden_size[i][0],
                    output_size=hidden_size[i][1],
                    hidden_size=h,
                    **kwargs
                )
            ))

        self.layers = nn.ModuleDict(OrderedDict(layers))

    def forward(self, x, y, hx=None, hy=None, choice=None):

        if self.training:

            if choice == None:
                return self.emb(x, y)
            elif self.mode == "Linear":
                return self.layers[f"layer_{choice}"](x, y)
            elif self.mode == "LSTM" or self.mode == "GRU":
                return self.layers[f"layer_{choice}"](x, y, hx, hy)

        else:

            if self.mode == "Linear":
                return self.layers[f"layer_{choice}"](x, y)
            elif self.mode == "LSTM" or self.mode == "GRU":
                return self.layers[f"layer_{choice}"](x, y, hx)

    def loss(self):

        sum = self.emb.loss()
        for layer in list(self.layers.values()):
            sum += layer.loss()

        return sum

    def inference(self, x) -> Tensor:

        k: str
        y: Tensor = x

        for k in list(self.layers.keys()):
            y = self.layers[k].f(y)

        y = self.layers[k].b(y)

        for k in list(self.layers.keys()).reverse():
            y = self.layers[k].decoder(y)

        return y


def test():

    inputs = torch.randint(1, 1000, (8, 2,))
    outputs = torch.randint(1, 1000, (8, 2,))
    model = ALNet("LSTM", [(512, 512), (256, 256)])
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    epochs = 5
    for _ in range(epochs):

        x_s1, y_s1 = model(inputs, outputs)
        x_s2, hx, y_s2, hy = model(x_s1, y_s1, choice=1)
        x_out, hx, y_out, hy = model(x_s2, y_s2, choice=2)
        loss = model.loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print(inputs.shape)
        # print(outputs.shape)
        print(x_out.size, y_out.size)
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
