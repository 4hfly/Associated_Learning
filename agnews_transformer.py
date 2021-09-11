# -*- coding: utf-8 -*-
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from transformer.encoder import TransformerEncoder
from transformer.encoder.utils import PositionalEncoding
from utils import *
import math
stop_words = set(stopwords.words('english'))


parser = argparse.ArgumentParser('AGNews Dataset for Transformer training')

# model param
parser.add_argument('--emb_dim', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--hid-dim', type=int,
                    help='hidden dimension', default=512)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.0001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=256)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=40)

# dir param
parser.add_argument('--save-dir', type=str,
                    default='data/ckpt/agnews_transformer.pt')

parser.add_argument('--pretrain-emb', type=str, default='glove')

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

train_features, train_mask = PadTransformer(clean_train_id, max_len)
test_features, test_mask = PadTransformer(clean_test_id, max_len)

print('train label num', len(train_label))
X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(
    train_features, train_mask, train_label, test_size=0.2, random_state=1)
X_test, mask_test, y_test = test_features, test_mask, test_label

train_data = TensorDataset(torch.from_numpy(
    X_train), torch.from_numpy(mask_train), torch.stack(y_train))
test_data = TensorDataset(torch.from_numpy(
    X_test), torch.from_numpy(mask_test), torch.stack(y_test))
valid_data = TensorDataset(torch.from_numpy(
    X_valid), torch.from_numpy(mask_valid), torch.stack(y_valid))

batch_size = args.batch_size

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


class TransformerForCLS(nn.Module):

    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, nhead, nlayers, class_num, dropout=0.5, pretrain=None
    ):

        super(TransformerForCLS, self).__init__()

        self.emb_dim = embedding_dim

        if pretrain == None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                pretrain, freeze=False, padding_idx=0)

        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.layernorm = nn.LayerNorm(embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        # NOTE: package 裡的 batch first = True
        # self.encoder = TransformerEncoder(
        #     embedding_dim, hidden_dim, nhead, nlayers, dropout)
        self.fc = nn.Linear(embedding_dim, class_num)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):

        # if mask == None:
        #     device = x.device
        #     mask = self._generate_square_subsequent_mask(x).to(device)

        x = self.embedding(x)
        # NOTE: positional encoding
        # 乘上 math.sqrt(self.emb_dim) 是模仿別人的，可以改。
        # remember to import math
        x = x * math.sqrt(self.emb_dim)
        x = self.pos_encoder(x)
        # NOTE: 假如是 transformer package 的，forward 參數只會有兩個，因此寫法是
        # output = self.encoder(x, src_key_padding_mask).sum(dim=1)
        # nn.Transformer 有三個參數，而 encoder 不需要 n*n 的 mask。
        output = self.encoder(x, src_mask, src_key_padding_mask).sum(dim=1)
        src_len = (src_key_padding_mask == 0).sum(dim=1)
        # fit the shape of output
        src_len = torch.stack((src_len,) * output.size(1), dim=1)
        output = output / src_len
        output = self.fc(output)

        return self.softmax(output)

    def _generate_square_subsequent_mask(self, sz: int):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        Shape: (sz, sz).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        return mask


torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.pretrain_emb != 'none':
    w = get_word_vector(vocab, emb=args.pretrain_emb)
else:
    w = None
nhead = 6
nlayers = 4
model = TransformerForCLS(args.vocab_size, args.emb_dim, args.hid_dim,
                          nhead, nlayers, class_num, pretrain=w)
model = model.to(device)
print('Transformer agnews model param num', get_n_params(model))
T = TransfomerTrainer(model, args.lr, train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir, is_al=False)
T.run(epoch=args.epoch)
T.eval()
