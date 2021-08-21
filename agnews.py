# -*- coding: utf-8 -*-
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classification.model import LSTMAL, EmbeddingAL
from utils import *

stop_words = set(stopwords.words('english'))


parser = argparse.ArgumentParser('AGNews Dataset for AL training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--label-emb', type=int,
                    help='label embedding dimension', default=128)
parser.add_argument('--l1-dim', type=int,
                    help='lstm1 hidden dimension', default=300)
parser.add_argument('--bridge-dim', type=int,
                    help='bridge function dimension', default=300)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=4)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)

# dir param
parser.add_argument('--save-dir', type=str, default='data/ckpt/agnews_al.pt')

args = parser.parse_args()


news_train = load_dataset('ag_news', split='train')
new_test = load_dataset('ag_news', split='test')

train_text = [b['text'] for b in news_train]
train_label = multi_class_process([b['label'] for b in news_train], 77)

test_text = [b['text'] for b in new_test]
test_label = multi_class_process([b['label'] for b in new_test], 77)

clean_train = [data_preprocessing(t) for t in train_text]
clean_test = [data_preprocessing(t) for t in test_text]

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length', max_len)

train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)

print('train label num', len(train_label))
X_train, X_valid, y_train, y_valid = train_test_split(
    train_features, train_label, test_size=0.2, random_state=1)
X_test, y_test = test_features, test_label

train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid))

batch_size = args.batch_size

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

# TODO: sample 沒用到的話，之後就刪掉嗎？
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

        batch_size = x.size(0)
        direction = 2

        emb_x, emb_y, = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        # print(self.embedding._t_prime.shape, self.embedding.y.shape)
        emb_loss = self.embedding.loss()

        layer_1_x, h1, layer_1_y = self.layer_1(emb_x.detach(), emb_y.detach())
        layer_1_x, layer_1_y = self.dropout(layer_1_x), self.dropout(layer_1_y)
        layer_1_loss = self.layer_1.loss()

        h, c = h1
        h = h.reshape(direction, batch_size, -1)
        h1 = (h.detach(), c.detach())

        layer_2_x, h2, layer_2_y = self.layer_2(
            layer_1_x.detach(), layer_1_y.detach(), h1)
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

# TODO: 這裡換成這樣就好
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb = EmbeddingAL((args.vocab_size, 77), (args.bridge_dim,
                                          args.bridge_dim), lin=args.one_hot_label)
l1 = LSTMAL(args.l1_dim, args.l1_dim, (args.bridge_dim,
                                       args.bridge_dim), dropout=0, bidirectional=True)
l2 = LSTMAL(2*args.l1_dim, args.l1_dim, (args.bridge_dim,
                                         args.bridge_dim), dropout=0, bidirectional=True)
model = SentAL(emb, l1, l2)
model = model.to(device)
print('AL agnews model param num', get_n_params(model))
T = ALTrainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epoch=args.epoch)
T.eval()

# TODO: code 比較長，之後我會把它拆成幾個小 function，再從 main() 這邊 call，這樣可讀性比較高，ok 吧？
