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


parser = argparse.ArgumentParser('AGNews Dataset for Transformer training')

# model param
parser.add_argument('--emb_dim', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--hid-dim', type=int,
                    help='lstm hidden dimension', default=400)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=32)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)

# dir param
parser.add_argument('--save-dir', type=str,
                    default='data/ckpt/agnews_transformer.pt')

args = parser.parse_args()


news_train = load_dataset('ag_news', split='train')
new_test = load_dataset('ag_news', split='test')

# TODO: 這個也要加進 args 裡面。
class_num = 4

train_text = [b['text'] for b in news_train]
train_label = multi_class_process([b['label'] for b in news_train], class_num)

test_text = [b['text'] for b in new_test]
test_label = multi_class_process([b['label'] for b in new_test], class_num)

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


class TransformerForCLS(nn.Module):

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, nhead, nlayers, class_num, dropout=0.5
    ):

        super(TransformerForCLS, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        layers = nn.TransformerEncoderLayer(
            embedding_dim, nhead, hidden_dim, dropout)
        self.encoder = nn.TransformerEncoder(layers, nlayers)
        self.mask = None
        self.fc = nn.Linear(embedding_dim, class_num)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        # if mask == None:
        #     device = x.device
        #     if self.mask == None:
        #         mask = self._generate_square_subsequent_mask(x).to(device)
        #         self.mask = mask
        # else:
        #     self.mask = mask

        x = self.embedding(x)
        output = self.encoder(x, self.mask).mean(dim=1)
        output = self.fc(output)

        return self.softmax(output)

    def _generate_square_subsequent_mask(self, sz: int):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nhead = 6
nlayers = 6
model = TransformerForCLS(args.vocab_size, args.emb_dim, args.hid_dim,
                          nhead, nlayers, class_num)
model = model.to(device)
print('Transformer agnews model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir, is_rnn=False)
T.run(epochs=args.epoch)
T.eval()
