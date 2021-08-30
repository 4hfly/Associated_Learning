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

stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser('YelpFull Dataset for AL training')

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
parser.add_argument('--batch-size', type=int, help='batch-size', default=128)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--class-num', type=int, default=5)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/yelp_al.pt')

args = parser.parse_args()


news_train = load_dataset('yelp_review_full', split='train')
new_test = load_dataset('yelp_review_full', split='test')

class_num = args.class_num

train_text = [b['text'] for b in news_train]
train_label = multi_class_process([b['label'] for b in news_train], class_num)

test_text = [b['text'] for b in new_test]
test_label = multi_class_process([b['label'] for b in new_test], class_num)

clean_train = [data_preprocessing(t, True) for t in train_text]

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

clean_test = [data_preprocessing(t, True) for t in test_text]
new_clean_test=[]
new_test_label=[]
for i, c in enumerate(clean_test):
    if len(c) == 0 or c == '' or c == "":
        lst.append(i)
    else:
        new_clean_test.append(c)
        new_test_label.append(test_label[i])
clean_test = new_clean_test
test_label = new_test_label

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length', max_len)
max_len = 400
train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)

print('train label num', len(train_label))
X_train, X_valid, y_train, y_valid = train_test_split(
    train_features, train_label, test_size=0.2, random_state=1)

X_test, y_test = test_features, test_label

train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train,dim=0))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test,dim=0))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid,dim=0))

batch_size = args.batch_size

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


class ClsAL(nn.Module):

    def __init__(self, emb, l1, l2):

        super(ClsAL, self).__init__()

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

        return emb_loss, layer_1_loss, 2*layer_2_loss

    def short_cut_emb(self, x):

        left = self.embedding.f(x)
        left = left.mean(-1)
        right = self.embedding.bx(left)
        right = self.embedding.dy(right)
        return right

    def short_cut_lstm(self, x):

        left = self.embedding.f(x)
        left, hidden = self.layer_1.f(left)
        left = left[:,-1,:]
        right = self.layer_1.bx(left)
        right = self.layer_1.dy(right)
        right = self.embedding.dy(right)
        return right



torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb = EmbeddingAL((args.vocab_size, class_num), (args.word_emb,
                                                 args.label_emb), lin=args.one_hot_label)
l1 = LSTMAL(args.word_emb, args.label_emb, (args.l1_dim,
                                            args.l1_dim), dropout=0, bidirectional=True)
l2 = LSTMAL(2 * args.l1_dim, args.l1_dim, (args.bridge_dim,
                                           args.bridge_dim), dropout=0, bidirectional=True)
model = ClsAL(emb, l1, l2)
model.load_state_dict(torch.load('ckpt/yelp_allast.pth'))
model = model.to(device)
print('AL Yelp full model param num', get_n_params(model))
T = ALTrainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
# T.run(epoch=args.epoch)
# T.eval()
# T.model.load_state_dict(torch.load('ckpt/yelp_allast.pth'))
T.short_cut_emb()
T.short_cut_l1()
