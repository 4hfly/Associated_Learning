# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sentencepiece as spm


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


# NOTE: same as input_transpose function.
def pad_sents(sents, pad_token):
    """
    Pad list of sentences according to the longest sentence in the batch.
    The paddings should be at the end of each sentence.
    """
    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    sents_t = []
    for i in range(max_len):
        sents_t.append([sents[k][i] if len(sents[k]) >
                        i else pad_token for k in range(batch_size)])

    return sents_t


def batch_iter(data, batch_size, shuffle=False):
    """
    Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def read_corpus(file_path, lang):
    """
    Read file, where each sentence is dilineated by a `\n`.
    """
    data = []
    # TODO: check the file path of tokenizer models.
    model_file = f'data/tokenizer/spm/{lang}/bpe.model'
    sp = spm.SentencePieceProcessor(model_file=model_file)

    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            subword_tokens = sp.encode_as_pieces(line)
            # TODO: do we always append <s> and </s> to the sentences?
            # subword_tokens = ['<s>'] + subword_tokens + ['</s>']
            data.append(subword_tokens)

    return data


# NOTE: word-level tokenizer
def read_corpus_by_word(file_path, lang):

    data = []

    for line in open(file_path):
        sent = line

        # TODO: define a tokenizer.
        # sent = tkr.encode(line)

        data.append(sent)

    return data
