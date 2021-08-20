from datasets import load_dataset
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import re
import string
from collections import Counter
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from classification.model import EmbeddingAL, LSTMAL
from utils import *
import sys

go_train = load_dataset('go_emotions', split='train')
go_val = load_dataset('go_emotions', split='validation')
go_test = load_dataset('go_emotions', split='test')

train_text = [b['text'] for b in go_train]
train_label = multi_label_process([b['labels'] for b in go_train])

val_text = [b['text'] for b in go_val]
val_label = multi_label_process([b['labels'] for b in go_val])

test_text = [b['text'] for b in go_test]
test_label = multi_label_process([b['labels'] for b in go_test])

clean_train = [data_preprocessing(t) for t in train_text]
clean_val = [data_preprocessing(t) for t in val_text]
clean_test = [data_preprocessing(t) for t in test_text]

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)
clean_val_id = convert2id(clean_val, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length',max_len)

raise Exception()

train_features = Padding(clean_train_id, max_len)
val_features = Padding(clean_val_id, max_len)
test_features = Padding(clean_test_id, max_len)

X_train, X_valid, y_train, y_valid = train_test_split(train_features, np.array(train_label), test_size=0.2, random_state=1)
X_test, y_test = test_features, np.array(test_label)
print(y_train.shape)

train_data = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))

batch_size = 16

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()


class SentAL(nn.Module):
    def __init__(self, emb, l1, l2):
        super(SentAL, self).__init__()
        self.embedding = emb
        self.layer_1 = l1
        self.layer_2 = l2
        self.dropout = nn.Dropout(0.1)
    def forward(self, x, y):

        batch_size=x.size(0)
        direction = 2

        emb_x, emb_y, = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        # print(self.embedding._t_prime.shape, self.embedding.y.shape)
        emb_loss = self.embedding.loss() 

        layer_1_x, h1 , layer_1_y = self.layer_1(emb_x.detach(), emb_y.detach())
        layer_1_x, layer_1_y = self.dropout(layer_1_x), self.dropout(layer_1_y)
        layer_1_loss = self.layer_1.loss()
        
        h,c = h1
        h = h.reshape(direction, batch_size, -1)
        h1 = (h.detach(),c.detach())
        
        layer_2_x, h2, layer_2_y = self.layer_2(layer_1_x.detach(), layer_1_y.detach(), h1)
        layer_2_loss = self.layer_2.loss()

        return emb_loss, layer_1_loss, layer_2_loss


torch.cuda.empty_cache()

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

vocab_size = len(vocab)+1

emb = EmbeddingAL((vocab_size, 77), (300, 300))
l1 = LSTMAL(300, 300, (300,300), dropout=0, bidirectional=True)
l2 = LSTMAL(600, 300, (300,300), dropout=0, bidirectional=True)
model = SentAL(emb, l1, l2)
model = model.to(device)
print('AL banking77 model param num', get_n_params(model))
T = ALTrainer(model, 0.001, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, save_dir = 'ckpt/banking77.al.pt')
T.run(epoch=10)
T.eval()