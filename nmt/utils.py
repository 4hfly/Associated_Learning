import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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

