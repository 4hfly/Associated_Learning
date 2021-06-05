import os
import torch
import pickle
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tqdm import tqdm

class Corpus(object):
    def __init__(self, path, ids=True):
        # self.dictionary = Dictionary()
        self.ids = ids 
        self.tokenizer = Tokenizer(BPE())
        self.tokenizer.model = BPE('data/vocab.json', 'data/merges.txt')
        self.train = self.tokenize(path)
        

    def tokenize(self, path):
        # assert os.path.exists(path)
        all_data = []
        if self.ids == False:
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in tqdm(lines, total=len(lines)):
                    line = line + "[SEP]"
                    encoding = self.tokenizer.encode(line)
                    ids = encoding.ids
                    all_data.append(torch.LongTensor(ids))
            with open(path+'.ids', 'w') as f:
                for data in all_data:
                    for i in data:
                        f.write(f'{i} ')
                    # f.write('\n'):d

            return all_data
        else:
            all_data = []
            with open(path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.split()
                    all_data = all_data + line
            tokens = torch.LongTensor(len(all_data))
            print('total tokens',len(all_data))
            for i, t in enumerate(all_data):
                tokens[i] = int(t)

            return tokens
            


# C = Corpus('data/pretrain/test.en', False)

