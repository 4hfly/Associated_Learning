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

os.environ['WANDB_SILENT'] = 'true'
stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser('Dbpedia Dataset for LSTM training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--label-emb', type=int,
                    help='label embedding dimension', default=128)
parser.add_argument('--l1-dim', type=int,
                    help='lstm1 hidden dimension', default=350)
parser.add_argument('--bridge-dim', type=int,
                    help='bridge function dimension', default=300)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=128)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--class-num', type=int, default=14)

parser.add_argument('--act', type=str,
                    default='tanh')
parser.add_argument('--pretrain-emb', type=str, default='glove')
parser.add_argument('--max-len', type=int, default=500)
parser.add_argument('--random-label', type=bool, default=False)
# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/dbpedia.pt')

args = parser.parse_args()

news_train = load_dataset('dbpedia_14', split='train')
new_test = load_dataset('dbpedia_14', split='test')

class_num = args.class_num

train_text = [b['content'] for b in news_train]
train_label = [b['label'] for b in news_train]
test_text = [b['content'] for b in new_test]
test_label = [b['label'] for b in new_test]

if args.random_label:
    train_label = multi_class_process([random.randint(0,19) for _ in range(len(train_text))], class_num)
    args.save_dir = args.save_dir[:-2]+'.rand.pt'
    print('This is a random label test')

clean_train = [data_preprocessing(t, True) for t in train_text]

clean_test = [data_preprocessing(t, True) for t in test_text]

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length', max_len)
max_len = args.max_len

train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)


X_train, X_valid, y_train, y_valid = train_test_split(
    train_features, train_label, test_size=0.2, random_state=1)

print('dataset information:')
print('=====================')
print('train size', len(X_train))
print('valid size', len(X_valid))
print('test size', len(test_features))
print('=====================')
X_test, y_test = test_features, test_label

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
        if pretrain == None:
            self.embedding = nn.Embedding.from_pretrain(pretrain, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
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


act = get_act(args)

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
print('DBpedia LSTM model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epochs=args.epoch)
T.eval()
