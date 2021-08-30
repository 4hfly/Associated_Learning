# -*- coding: utf-8 -*-
# TODO: 有些沒用到的 lib 我之後會拿掉喔。
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classification.model import EmbeddingAL, TransformerEncoderAL
from utils import *

stop_words = set(stopwords.words('english'))

parser = argparse.ArgumentParser(
    'Banking77 Dataset for AL Transformer training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--label-emb', type=int,
                    help='label embedding dimension', default=128)
parser.add_argument('--l1-dim', type=int,
                    help='lstm1 hidden dimension', default=300)
parser.add_argument('--l2-dim', type=int,
                    help='layer 2 hidden dimension', default=300)
parser.add_argument('--nhead', type=int, default=6)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=16)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=50)

# dir param
parser.add_argument('--save-dir', type=str,
                    default='data/ckpt/banking77_al_trans.pt')

args = parser.parse_args()


bank_train = load_dataset('banking77', split='train')
bank_test = load_dataset('banking77', split='test')

train_text = [b['text'] for b in bank_train]
train_label = multi_class_process([b['label'] for b in bank_train], 77)

test_text = [b['text'] for b in bank_test]
test_label = multi_class_process([b['label'] for b in bank_test], 77)

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

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()


class CLSAL(nn.Module):

    def __init__(self, emb, l1, l2):
        super(CLSAL, self).__init__()
        self.embedding = emb
        self.layer_1 = l1
        self.layer_2 = l2
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y):

        emb_x, emb_y = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        emb_loss = self.embedding.loss()

        layer_1_x, layer_1_y = self.layer_1(emb_x.detach(), emb_y.detach())
        layer_1_loss = self.layer_1.loss()

        layer_2_x, layer_2_y = self.layer_2(
            layer_1_x.detach(), layer_1_y.detach())
        layer_2_loss = self.layer_2.loss()

        return emb_loss, layer_1_loss, layer_2_loss


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emb = EmbeddingAL((args.vocab_size, 77), (args.word_emb,
                                          args.label_emb), lin=args.one_hot_label)
l1 = TransformerEncoderAL(
    (args.word_emb, args.label_emb), args.nhead, args.l1_dim)
l2 = TransformerEncoderAL((args.l1_dim, args.l1_dim), args.nhead, args.l2_dim)
model = CLSAL(emb, l1, l2)
model = model.to(device)
print('AL Transformer banking77 model param num', get_n_params(model))
T = TransfomerTrainer(model, args.lr, train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epoch=args.epoch)
T.eval()

# TODO: code 比較長，之後我會把它拆成幾個小 function，再從 main() 這邊 call，這樣可讀性比較高，ok 吧？
