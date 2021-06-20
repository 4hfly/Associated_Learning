# -*- coding: utf-8 -*-

import math

import torch
import pickle
import numpy as np
from torch.autograd import Variable
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel
from model import RNNModel 


class LMProb():

    def __init__(self, model_path, dict_path):
         
        self.model = RNNModel(25000, 200, 256, 2, 0.2)
        self.model.load_state_dict(torch.load(model_path))


        self.tkr = Tokenizer(BPE())
        self.tkr.normalizer = Sequence([
            NFKC(),
            Lowercase()
            ])
        self.tkr.pre_tokenizer = ByteLevel()
        self.tkr.model = BPE(dict_path+'vocab.json', dict_path+'merges.txt')

    def get_prob(self, words, verbose=False):
        
        inp = self.tkr.encode(words)
        try:
            pad_idx = inp.ids.find(0)
        except:
            pad_idx = len(inp.ids)
        list_ids = inp.ids

        hidden = self.model.init_hidden(1)
        log_probs = []
        for i in range(pad_idx):
            word_ids = Variable(torch.LongTensor([inp.ids[i]]).unsqueeze(1))
            output, hidden = self.model(word_ids, hidden)
            word_weights = output.squeeze().data.exp()
            w = word_weights[list_ids[i]]
            prob = w / word_weights.sum()
            log_probs.append(math.log(prob))
            
        if verbose:
            print('\n  => sum_prob = {:.4f}'.format(sum(log_probs)))

        return sum(log_probs) / math.sqrt(len(log_probs))


if __name__ == '__main__':
    words = ['we', 'have', 'told', 'that', 'this', 'will']
    words = ' '.join(words)
    print('words: ', words)
    words = "This is a simple sentence ."
    lmprob = LMProb('model_ckp/lm.en.e40.pth', 'data/')
    norm_prob = lmprob.get_prob(words, verbose=True)
    print('\n  => norm_prob = {:.4f}'.format(norm_prob))
