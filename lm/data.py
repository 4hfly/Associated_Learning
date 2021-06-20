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
        self.tokenizer.model = BPE('data/fr/vocab.json', 'data/fr/merges.txt')
        self.train = self.tokenize(path)
        self.valid = self.tokenize('data/pretrain/pretrain.valid.fr.ids') 

    def tokenize(self, path):
        # assert os.path.exists(path)
        all_data = []
        if self.ids == False:
            # file is not in word_id format. and turn all data into word_id, and save file.
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
            # if file is in word_id format, then concat all words as a very long tensor [15, 432, 2, 100,8976,342 ........]
            # 如果已經轉成id格式，就把data讀起來接成一個超長的tensor
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
            


# C = Corpus('data/pretrain/pretrain.train.fr', False)

