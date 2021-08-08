#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import ByteLevelBPETokenizer

DATA = [
    "data/wmt14/commoncrawl/commoncrawl.fr-en.fr",
    "data/wmt14/europarl_v7/europarl-v7.fr-en.fr",
    "data/wmt14/giga/giga-fren.release2.fixed.fr",
    "data/wmt14/news-commentary/news-commentary-v9.fr-en.fr",
    "data/wmt14/un/undoc.2000.fr-en.fr"
]


class LabelSmoothingLoss(nn.Module):
    """
    label smoothing
    Code adapted from OpenNMT-py
    """

    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        # -1 for pad, -1 for gold-standard word
        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x tgt_vocab_size
        target (LongTensor): batch_size
        """
        # (batch_size, tgt_vocab_size)
        output = output.float()
        # target = target.float()
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)
        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss


def preprocess_text():

    for file in DATA:
        with open(file, "r", encoding="utf8") as f:
            data = f.readlines()

        with open(f"{file}.shell", "w", encoding="utf8") as f:
            for s in data:
                f.write(f"<s> {s.strip()} </s>\n")


def test():

    def bpe(
        input,
        lang="fr",
        files=[
            "data/tokenizer/fr/vocab.json",
            "data/tokenizer/fr/merges.txt"
        ]
    ):

        tokenizer = ByteLevelBPETokenizer(lang=lang, files=files)
        encoded = tokenizer.encode(input)
        decoded = tokenizer.decode(encoded)
        return encoded, decoded

    encoded, decoded = bpe("Bonjour, vous tous! Comment ça va 😁?")
    print(encoded.tokens)
    print(decoded)


if __name__ == "__main__":
    test()
