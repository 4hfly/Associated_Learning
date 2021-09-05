import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classification.model import EmbeddingAL, LSTMAL
from utils import *

import pandas as pd

import os
os.environ["WANDB_SILENT"] = "true"

parser = argparse.ArgumentParser('IMDB Dataset for AL training')

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

parser.add_argument('--pretrain-emb', type=str, help='pretrained word embedding: glove or fasttest', default='glove')
parser.add_argument('--data-position', type=str, help='shuffle or sort', default='shuffle')
# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=64)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)

# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/imdb.al.pt')

parser.add_argument('--act', type=str,
                    default='tanh')

args = parser.parse_args()
df = pd.read_csv('IMDB_Dataset.csv')
df['cleaned_reviews'] = df['review'].apply(data_preprocessing)
corpus = [word for text in df['cleaned_reviews'] for word in text.split()]

vocab = create_vocab(corpus, args.vocab_size)

print('vocab size',len(vocab))

clean_train = [data_preprocessing(t) for t in corpus]
clean_train_id = convert2id(clean_train, vocab)

df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
train_label = df['sentiment'].tolist()
train_label = multi_class_process(train_label, 2)

train_features = Padding(clean_train_id, 400)
shuf=True
if args.data_position == 'sort':
    shuf = False
X_train, X_remain, y_train, y_remain = train_test_split(train_features, train_label, test_size=0.2, random_state=1, shuffle=shuf)
X_valid, X_test, y_valid, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=1)

# create tensor dataset
train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid))

batch_size = args.batch_size

if args.data_position == 'sort':
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
else:
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


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

# device = 'cpu'

# Instantiate the model w/ hyperparams
if args.pretrain_emb == 'none':
    emb = EmbeddingAL((args.vocab_size, 2), (args.bridge_dim, args.bridge_dim), lin=args.one_hot_label, act=act)
else:
    w = get_word_vector(vocab, emb=args.pretrain_emb)
    emb = EmbeddingAL((args.vocab_size, 2), (args.bridge_dim, args.bridge_dim), lin=args.one_hot_label, pretrained=w,act=act)

l1 = LSTMAL(args.l1_dim, args.l1_dim, (args.bridge_dim, args.bridge_dim), dropout=0, bidirectional=True, act=act)
l2 = LSTMAL(2*args.l1_dim, args.l1_dim, (args.bridge_dim, args.bridge_dim), dropout=0, bidirectional=True, act=act)
model = SentAL(emb, l1, l2)
model = model.to(device)
print('AL IMDB model param num', get_n_params(model))
T = ALTrainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epoch=args.epoch)
T.eval()
T.tsne_()