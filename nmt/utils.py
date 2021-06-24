import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

class NMTData(Dataset):
    def __init__(self, en_data, fr_data, en_tkr, fr_tkr):
        with open(en_data) as f:
            lines = f.readlines()
            en_sents = [l.replace('\n', '') for l in lines]

        with open(fr_data) as f:
            lines = f.readlines()
            fr_sents = [l.replace('\n', '') for l in lines]
        
        self.en_data = [en_tkr.encode(s).ids for s in en_sents]
        self.fr_data = [fr_tkr.encode(s).ids for s in fr_sents]

        assert len(en_sents) == len(fr_sents)
    def __len__(self):
        return len(fr_sents)

    def  __getitem__(self, idx):
        return torch.tensor(en_data[idx]), torch.tensor(fr_data[idx])

def collate(batch):
    en_sents = [e for e,f in batch]
    fr_sents = [f for e,f in batch]
    en_pad_sents = pad_sequence(en_sents, batch_first=True)
    fr_pad_sents = pad_sequence(fr_sents, batch_first=True)
    return en_pad_sents, fr_pad_sents


def word2id(sents, tkr):
    if type(sents[0]) == list:
        return [tkr.encode(s).ids for s in sents]
    else:
        return tkr.encode(s).ids


def id2word(sents, tkr):
    if type(sents[0]) == list:
        return [tkr.decode(s).ids for s in sents]
    else:
        return tkr.decode(s).ids

class LabelSmoothingLoss(nn.Module):
    """
    label smoothing
    Code adapted from OpenNMT-py
    """
    def __init__(self, label_smoothing, tgt_vocab_size, padding_idx=0):
        assert 0.0 < label_smoothing <= 1.0
        self.padding_idx = padding_idx
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)  # -1 for pad, -1 for gold-standard word
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
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)

        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss