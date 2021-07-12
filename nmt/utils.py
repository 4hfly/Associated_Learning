import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

class NMTData(Dataset):
    def __init__(self, en_data, fr_data, en_tkr, fr_tkr, task="en2fr"):
        with open(en_data) as f:
            lines = f.readlines()
            en_sents = [l.replace('\n', '') for l in lines]

        with open(fr_data) as f:
            lines = f.readlines()
            fr_sents = [l.replace('\n', '') for l in lines]

        self.task = task

        self.en_data = [en_tkr.encode(s).ids for s in en_sents]
        self.fr_data = [fr_tkr.encode(s).ids for s in fr_sents]

        assert len(en_sents) == len(fr_sents)
    def __len__(self):
        return len(self.fr_data)

    def  __getitem__(self, idx):
        en_sent = self.en_data[idx]
        fr_sent = self.fr_data[idx]
        if self.task == "en2fr":
            fr_sent = [1] + fr_sent + [2]
            return torch.tensor(en_sent), torch.tensor(fr_sent)
        else:
            en_sent = [1] + en_sent + [2]
            return torch.tensor(fr_sent), torch.tensor(en_sent)
            
def collate(batch):
    src_sents = [e for e,f in batch]
    tgt_sents = [f for e,f in batch]
    src_pad_sents = pad_sequence(src_sents, batch_first=True)
    tgt_pad_sents = pad_sequence(tgt_sents, batch_first=True)
    # tgt_pad_sents = torch.stack(tgt_sents)
    return src_pad_sents, tgt_pad_sents


def word2id(sents, tkr):
    if type(sents[0]) == list:
        return [tkr.encode(s).ids for s in sents]
    else:
        return tkr.encode(sents).ids


def id2word(sents, tkr):
    if type(sents[0]) == list:
        return [tkr.decode(s).ids for s in sents]
    else:
        return tkr.decode(sents).ids

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
        # print(output.shape)
        # print(target.shape)
        # (batch_size, tgt_vocab_size)
        output = output.float()
        # target = target.float()
        true_dist = self.one_hot.repeat(target.size(0), 1)

        # fill in gold-standard word position with confidence value
        true_dist.scatter_(1, target.unsqueeze(-1), self.confidence)

        # fill padded entries with zeros
        true_dist.masked_fill_((target == self.padding_idx).unsqueeze(-1), 0.)
        # print('true dist',true_dist)
        # print('output', output)
        loss = -F.kl_div(output, true_dist, reduction='none').sum(-1)

        return loss


def batch_iter(data, batch_size, shuffle=False):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        # if i ==0:
        #     print(examples[0])
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def read_corpus(file_path, source):
    data = []
    for line in open(file_path):
        sent = line.strip().replace('\n', '')
        # only append <s> and </s> to the target sentence
        # if source == 'tgt':
        #     sent = "<s> " + sent + " </s>"
        data.append(sent)

    return data

def input_transpose(sents, pad_token):
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) > i else pad_token for k in range(batch_size)])

    return sents_t


def to_input_tensor(sents, tkr, device: torch.device, tgt=False) -> torch.Tensor:

    if tgt:
        sents = [" "+s for s in sents]
        word_ids = [[1]+tkr.encode(s).ids+[2] for s in sents]
    else:
        word_ids = [tkr.encode(s).ids for s in sents] 

    sents_len = [len(s) for s in word_ids]
    sents_t = input_transpose(word_ids, 0)

    sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)

    return sents_var, sents_len




# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.normalizers import Lowercase, NFKC, Sequence
# from tokenizers.pre_tokenizers import ByteLevel
# from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# def get_tkr(dp):
#     tgt_tkr = Tokenizer(BPE())
#     tgt_tkr.normalizer = Sequence([
#         NFKC(),
#         Lowercase()
#         ])
#     tgt_tkr.pre_tokenizer = ByteLevel()
#     tgt_tkr.decoder = ByteLevelDecoder()
#     tgt_tkr.model = BPE(dp+'/vocab.json', dp+'/merges.txt')
#     return tgt_tkr
# D = NMTData('./data/train/en-corpus/train.data.en.test', './data/train/fr-corpus/train.data.fr.test', get_tkr('./data/tkr/en'), get_tkr('./data/tkr/fr'))
# L = DataLoader(D, batch_size=8, collate_fn = collate)
# for d in L:
#     print(d)
#     break