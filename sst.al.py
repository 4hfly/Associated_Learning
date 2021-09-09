# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from one_layer import L1_ALTrainer
from classification.model import LSTMAL, EmbeddingAL
from utils import *
import os
import warnings
warnings.simplefilter("ignore")
os.environ["WANDB_SILENT"] = "true"
stop_words = set(stopwords.words('english'))


parser = argparse.ArgumentParser('SST Dataset for AL training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--label-emb', type=int,
                    help='label embedding dimension', default=128)
parser.add_argument('--l1-dim', type=int,
                    help='layer 1 hidden dimension', default=300)
parser.add_argument('--l2-dim', type=int,
                    help='layer 2 hidden dimension', default=300)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=64)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--class-num', type=int, default=2)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/sst.al.pt')

parser.add_argument('--act', type=str,
                    default='tanh')
parser.add_argument('--pretrain-emb', type=str, default='glove')

args = parser.parse_args()


news_train = load_dataset('glue', 'sst2', split='train')
news_val = load_dataset('glue', 'sst2', split='validation')
new_test = load_dataset('glue', 'sst2', split='test')

# TODO: 這個也要加進 args 裡面。
class_num = args.class_num

train_text = [b['sentence'] for b in news_train]
train_label = multi_class_process([b['label'] for b in news_train], class_num)

val_text = [b['sentence'] for b in news_val]
val_label = multi_class_process([b['label'] for b in news_val], class_num)

test_text = [b['sentence'] for b in new_test]
test_label = multi_class_process([b['label'] for b in new_test], class_num)

clean_train = [data_preprocessing(t) for t in train_text]
clean_val = [data_preprocessing(t) for t in val_text]
clean_test = [data_preprocessing(t) for t in test_text]

vocab = create_vocab(clean_train)

clean_train_id = convert2id(clean_train, vocab)
clean_val_id = convert2id(clean_val, vocab)
clean_test_id = convert2id(clean_test, vocab)

max_len = max([len(s) for s in clean_train_id])
print('max seq length', max_len)

train_features = Padding(clean_train_id, max_len)
test_features = Padding(clean_test_id, max_len)
val_features = Padding(clean_val_id, max_len)

X_train, X_valid, y_train, y_valid = train_features, val_features, train_label, val_label
X_test, y_test = test_features, test_label

print('dataset information:')
print('=====================')
print('train size', len(X_train))
print('valid size', len(X_valid))
print('test size', len(test_features))
print('=====================')

train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid))

batch_size = args.batch_size

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


class CLSAL(nn.Module):

    def __init__(self, emb, l1, l2):

        super(CLSAL, self).__init__()

        self.embedding = emb
        self.layer_1 = l1
        # self.layer_2 = l2
        self.dropout = nn.Dropout(0.3)
        self.count = 0

    def forward(self, x, y):
        self.count += 1
        batch_size = x.size(0)
        direction = 2

        emb_x, emb_y = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        # print(self.embedding._t_prime.shape, self.embedding.y.shape)
        emb_loss = self.embedding.loss()

        layer_1_x, h1, layer_1_y = self.layer_1(emb_x.detach(), emb_y.detach())
        layer_1_x, layer_1_y = self.dropout(layer_1_x), self.dropout(layer_1_y)
        layer_1_loss = self.layer_1.loss()

        return emb_loss, layer_1_loss


act = get_act(args)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.vocab_size = len(vocab)

if args.pretrain_emb == 'none':
    emb = EmbeddingAL((args.vocab_size, class_num), (args.word_emb,
                                                     args.label_emb), lin=args.one_hot_label, act=act)
else:
    w = get_word_vector(vocab, emb=args.pretrain_emb)
    emb = EmbeddingAL((args.vocab_size, class_num), (args.word_emb,
                                                     args.label_emb), lin=args.one_hot_label, pretrained=w, act=act)

l1 = LSTMAL(args.word_emb, args.label_emb, (args.l1_dim,
                                            args.l1_dim), dropout=0., bidirectional=True, act=act)
l2 = LSTMAL(2 * args.l1_dim, args.l1_dim, (args.l2_dim,
                                           args.l2_dim), dropout=0., bidirectional=True, act=act)
model = CLSAL(emb, l1, l2)
model = model.to(device)

count_parameters(model)
input('wait for input enter')
T = L1_ALTrainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epoch=args.epoch)
T.write_pred()
# T.tsne_()
