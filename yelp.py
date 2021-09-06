# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classification.model import LSTMAL, EmbeddingAL
from utils import *
import os

os.environ["WANDB_SILENT"] = "true"
stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser('YelpFull Dataset for LSTM training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--l1-dim', type=int,
                    help='lstm1 hidden dimension', default=400)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)
parser.add_argument('--pretrain-emb', type=str, default='glove')
# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=32)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--class-num', type=int, default=5)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/yelp.pt')

args = parser.parse_args()


news_train = load_dataset('yelp_review_full', split='train')
new_test = load_dataset('yelp_review_full', split='test')

# TODO: 這個也要加進 args 裡面。
class_num = args.class_num

train_text = [b['text'] for b in news_train]
train_label = multi_class_process([b['label'] for b in news_train], class_num)

test_text = [b['text'] for b in new_test]
test_label = multi_class_process([b['label'] for b in new_test], class_num)

clean_train = [data_preprocessing(t, True) for t in train_text]
clean_test = [data_preprocessing(t, True) for t in test_text]

lst = []

new_clean_train = []
new_train_label=[]
for i, c in enumerate(clean_train):
    if len(c) == 0 or c == '' or c == "":
        lst.append(i)
    else:
        new_clean_train.append(c)
        new_train_label.append(train_label[i])
clean_train = new_clean_train
train_label=new_train_label

clean_test = [data_preprocessing(t) for t in test_text]
new_clean_test=[]
new_test_label=[]
for i, c in enumerate(clean_test):
    if len(c) == 0 or c == '' or c == "":
        lst.append(i)
    else:
        new_test_label.append(test_label[i])
        new_clean_test.append(c)
clean_test = new_clean_test
test_label=new_test_label

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length', max_len)
max_len=400
train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)

X_train, X_valid, y_train, y_valid = train_test_split(
    train_features, train_label, test_size=0.2, random_state=1)
X_test, y_test = test_features, test_label

print('dataset information:')
print('=====================')
print('train size', len(X_train))
print('valid size', len(X_valid))
print('test size', len(test_features))
print('=====================')
train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train,dim=0))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test,dim=0))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid,dim=0))

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
        if pretrain is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.pretrain_emb == 'none': 
    model = Cls(args.vocab_size, args.word_emb, args.l1_dim, 2, args.class_num)
else:
    w = get_word_vector(vocab, emb=args.pretrain_emb)
    model = Cls(args.vocab_size, args.word_emb, args.l1_dim, 2, args.class_num, pretrain=w)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)
print('Yelp LSTM model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epochs=args.epoch)
T.eval()
T.eval()
T.tsne_()
