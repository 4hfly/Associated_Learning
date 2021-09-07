# -*- coding: utf-8 -*-
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


parser = argparse.ArgumentParser('AGNews Dataset for Transformer training')

# model param
parser.add_argument('--pretrain-emb', type=str, default='glove')
parser.add_argument('--emb-dim', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--label-dim', type=int,
                    help='label embedding dimension', default=128)
parser.add_argument('--hid-dim', type=int,
                    help='hidden dimension', default=510)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)
parser.add_argument('--act', type=str, default='tanh')

# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=32)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)

# dir param
parser.add_argument('--save-dir', type=str,
                    default='ckpt/agnews_transformer.al.pt')

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

    def __init__(self, emb, l1, l2):

        super(TransformerForCLS, self).__init__()
        self.embedding = emb
        self.layer_1 = l1
        self.layer_2 = l2
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, src_mask=None, src_key_padding_mask=None):

        # if mask == None:
        #     device = x.device
        #     mask = self._generate_square_subsequent_mask(x).to(device)
        emb_x, emb_y = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        emb_loss = self.embedding.loss()

        layer_1_x, layer_1_y = self.layer_1(
            emb_x.detach(), emb_y.detach(), src_mask, src_key_padding_mask)
        layer_1_loss = self.layer_1.loss()

        layer_2_x, layer_2_y = self.layer_2(
            layer_1_x.detach(), layer_1_y.detach(), src_mask, src_key_padding_mask)
        layer_2_loss = self.layer_2.loss()

        return emb_loss, layer_1_loss, layer_2_loss

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

act = get_act(args)
if args.pretrain_emb == 'none':
    emb = EmbeddingAL((args.vocab_size, class_num), (args.emb_dim,
                                                     args.label_dim), lin=args.one_hot_label, act=act)
else:
    w = get_word_vector(vocab, emb=args.pretrain_emb)
    emb = EmbeddingAL((args.vocab_size, class_num), (args.emb_dim,
                                                     args.label_dim), lin=args.one_hot_label, pretrained=w, act=act)
# TODO: 這裡 y 的 hidden size 也許需要再調整大小。
nhead = 6
l1 = TransformerEncoderAL((args.emb_dim, args.label_dim), nhead,
                          args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)
l2 = TransformerEncoderAL((args.emb_dim, args.hid_dim), nhead,
                          args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)

model = TransformerForCLS(emb, l1, l2)
model = model.to(device)
print('Transformer AL agnews model param num', get_n_params(model))
T = TransfomerTrainer(model, args.lr, train_loader=train_loader,
                      valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir, is_al=True)
T.run(epoch=args.epoch)
T.eval()
