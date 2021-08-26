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
import torch.nn.functional as F

from classification.model import EmbeddingAL, LSTMAL
from utils import *
import sys

import argparse

parser = argparse.ArgumentParser('Banking77 Dataset for LSTM training')

# model param
parser.add_argument('--emb-dim', type=int, help='word embedding dimension', default=300)
parser.add_argument('--hid-dim', type=int, help='lstm1 hidden dimension', default=400)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=16)
parser.add_argument('--epoch', type=int, default=50)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/banking77.pt')

args = parser.parse_args()



bank_train = load_dataset('banking77', split='train')
bank_test = load_dataset('banking77', split='test')

train_text = [b['text'] for b in bank_train]
train_label = multi_class_process([b['label'] for b in bank_train], 77)
label_dist = Counter([b['label'] for b in bank_train])
test_text = [b['text'] for b in bank_test]
test_label = multi_class_process([b['label'] for b in bank_test], 77)

clean_train = [data_preprocessing(t) for t in train_text]
clean_test = [data_preprocessing(t) for t in test_text]

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length',max_len)

train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)

print('train label num', len(train_label))
X_train, X_valid, y_train, y_valid = train_test_split(train_features, train_label, test_size=0.2, random_state=1)
X_test, y_test = test_features, test_label

train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid))

batch_size = args.batch_size

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

class sentimentLSTM(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5, class_num=77):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        # Embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        # Dropout layer
        self.dropout = nn.Dropout(0.3)
        # Linear and sigmoid layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim*2, 400),
            nn.ReLU(),
            nn.Linear(400, class_num)
        )
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out[:,-1,:]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        sig_out = self.softmax(out)
        return sig_out, hidden

torch.cuda.empty_cache()

is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

dist = []

for i in range(len(label_dist)):
    dist.append(label_dist[i])

m = sum(dist)
print(m)
dist = [x/m for x in dist]
dist = [1/x for x in dist]
dist = torch.tensor(dist)
dist = F.normalize(dist,dim=0)

model = sentimentLSTM(args.vocab_size, args.emb_dim, args.hid_dim, 2)
model = model.to(device)
print('LSTM banking77 model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader, valid_loader=valid_loader, test_loader=test_loader, save_dir = args.save_dir, loss_w=dist)
T.run(epochs=args.epoch)
T.eval()
