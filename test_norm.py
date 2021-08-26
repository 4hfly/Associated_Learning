# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from utils import *

stop_words = set(stopwords.words('english'))


parser = argparse.ArgumentParser('AGNews Dataset for AL training')

# model param
parser.add_argument('--emb_dim', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--hid-dim', type=int,
                    help='lstm hidden dimension', default=400)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=16)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/agnews.pt')

args = parser.parse_args()


class_num = 4

train_features = [np.array([1,2,3]) for i in range(400)] + [np.array([3,4,5]) for i in range(400)] + [np.array([5,6,7]) for i in range(400)]
test_features = train_features

train_label = [torch.tensor([1,0,0,0])]*400 + [torch.tensor([0,1,0,0])]*400 + [torch.tensor([0,0,1,0])]*400
test_label = train_label

# train_features = Padding(clean_train_id, max_len)
# test_features = Padding(clean_test_id, max_len)

print('train label num', len(train_label))
X_train, X_valid, y_train, y_valid = train_test_split(
    train_features, train_label, test_size=0.2, random_state=1)
X_test, y_test = test_features, test_label

X_train = [torch.from_numpy(x) for x in X_train]
X_test = [torch.from_numpy(x) for x in X_test]
X_valid = [torch.from_numpy(x) for x in X_valid]



train_data = TensorDataset(torch.stack(X_train), torch.stack(y_train))
test_data = TensorDataset(torch.stack(X_test), torch.stack(y_test))
valid_data = TensorDataset(torch.stack(X_valid), torch.stack(y_valid))

batch_size = args.batch_size

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


class ClsAL(nn.Module):

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, n_layers, class_num, drop_prob=0.5
    ):

        super(ClsAL, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
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


torch.cuda.empty_cache()

# TODO: 這裡換成這樣就好
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ClsAL(args.vocab_size, args.emb_dim, args.hid_dim, 2, class_num)
model = model.to(device)
print('agnews lstm model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epochs=args.epoch)
T.eval()

# TODO: code 比較長，之後我會把它拆成幾個小 function，再從 main() 這邊 call，這樣可讀性比較高，ok 吧？
