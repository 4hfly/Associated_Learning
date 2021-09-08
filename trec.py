# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils import *
import os
os.environ["WANDB_SILENT"] = "true"
stop_words = set(stopwords.words('english'))


parser = argparse.ArgumentParser('TREC Dataset for LSTM training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--l1-dim', type=int,
                    help='lstm hidden dimension', default=400)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=32)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/trec.pt')
parser.add_argument('--pretrain-emb', type=str, default='glove')
parser.add_argument('--class-num', type=int, default=6)

args = parser.parse_args()


news_train = load_dataset('trec', split='train')
new_test = load_dataset('trec', split='test')

# TODO: 這個也要加進 args 裡面。
class_num = args.class_num

train_text = [b['text'] for b in news_train]
train_label = multi_class_process([b['label-coarse'] for b in news_train], class_num)

test_text = [b['text'] for b in new_test]
test_label = multi_class_process([b['label-coarse'] for b in new_test], class_num)

clean_train = [data_preprocessing(t, True) for t in train_text]
clean_test = [data_preprocessing(t, True) for t in test_text]

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length', max_len)

train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)

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


class Cls(nn.Module):
    
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, n_layers, class_num, drop_prob=0.1, pretrain=None
    ):

        super(Cls, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if pretrain == None:
            self.embedding = nn.Embedding.from_pretrain(pretrain, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        n_layers=2
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 400),
            nn.ReLU(),
            nn.Linear(400, class_num)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden


torch.cuda.empty_cache()

# TODO: 這裡換成這樣就好
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('current using device:', device)

if args.pretrain_emb == 'none': 
    model = Cls(args.vocab_size, args.word_emb, args.l1_dim, 2, args.class_num)
else:
    w = get_word_vector(vocab, emb=args.pretrain_emb)
    model = Cls(args.vocab_size, args.word_emb, args.l1_dim, 2, args.class_num, pretrain=w)

model = model.to(device)
print('TREC lstm model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epochs=args.epoch)
T.eval()