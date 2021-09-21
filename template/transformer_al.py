# -*- coding: utf-8 -*-
import argparse
import json

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from transformer.encoder import TransformerEncoder
from transformer.encoder.utils import PositionalEncoding
from classification.model import EmbeddingAL, TransformerEncoderAL
from utils import *

# TODO: 可以寫成一個 class 取代 parser。
# Parameters:
#   `pretrained`: str | None，不建議用 'none'
#   `activation`: str
CONFIG = {
    'Title': 'Template',
    'dataset': 'template',
    'Parameters': {
        'vocab_size': 30000,
        'pretrained': 'glove',
        'embedding_dim': 300,
        'label_dim': 128,
        'hidden_dim': 256,
        'nhead': 6,
        'nlayers': 2,
        'class_num': 2,
        'max_len': 256,
        'one_hot_label': True,
        'activation': 'tanh',
        'lr': 1e-3,
        'batch_size': 256,
        'epochs': 40,
        'ramdom_labe;': False
    },
    "Save_dir": 'ckpt/',
}


class TransformerForCLS(nn.Module):

    def __init__(self, emb, l1, l2, l3, l4, l5, l6):

        super(TransformerForCLS, self).__init__()
        self.embedding = emb
        self.layer_1 = l1
        self.layer_2 = l2
        # self.layer_3 = l3
        # self.layer_4 = l4
        # self.layer_5 = l5
        # self.layer_6 = l6
        # NOTE: still has some bugs.
        # self.layers = nn.ModuleList([copy.deepcopy(module) for _ in range(n - 1)])

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, y, src_mask=None, src_key_padding_mask=None):

        layer_loss = []
        # if mask == None:
        #     device = x.device
        #     mask = self._generate_square_subsequent_mask(x).to(device)
        emb_x, emb_y = self.embedding(x, y)
        emb_x, emb_y = self.dropout(emb_x), self.dropout(emb_y)
        emb_loss = self.embedding.loss()

        out_x, out_y = self.layer_1(
            emb_x.detach(), emb_y.detach(), src_mask, src_key_padding_mask
        )
        layer_loss.append(self.layer_1.loss())

        out_x, out_y = self.layer_2(
            out_x.detach(), out_y.detach(), src_mask, src_key_padding_mask
        )
        layer_loss.append(self.layer_2.loss())

        # out_x, out_y = self.layer_3(
        #     out_x.detach(), out_y.detach(), src_mask, src_key_padding_mask
        # )
        # layer_loss.append(self.layer_3.loss())

        # out_x, out_y = self.layer_4(
        #     out_x.detach(), out_y.detach(), src_mask, src_key_padding_mask
        # )
        # layer_loss.append(self.layer_4.loss())

        # out_x, out_y = self.layer_5(
        #     out_x.detach(), out_y.detach(), src_mask, src_key_padding_mask
        # )
        # layer_loss.append(self.layer_5.loss())

        # out_x, out_y = self.layer_6(
        #     out_x.detach(), out_y.detach(), src_mask, src_key_padding_mask
        # )
        # layer_loss.append(self.layer_6.loss())

        return emb_loss, layer_loss

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


def arg_parser():
    '''CONFIG 轉為 class 後可棄用'''

    t = CONFIG['Title']
    parser = argparse.ArgumentParser(f'{t} for Transformer training.')

    # NOTE: args 的 prog name 有些和慣用的不一樣。
    # model param
    parser.add_argument(
        '--vocab-size', type=int,
        help='vocab-size',
        default=CONFIG['Parameters']['vocab_size']
    )
    parser.add_argument(
        '--pretrained', type=str,
        default=CONFIG['Parameters']['pretrained']
    )
    parser.add_argument(
        '--emb-dim', type=int,
        help='word embedding dimension',
        default=CONFIG['Parameters']['embedding_dim']
    )
    parser.add_argument(
        '--label-dim', type=int,
        help='y(label) dimension',
        default=CONFIG['Parameters']['label_dim']
    )
    parser.add_argument(
        '--hid-dim', type=int,
        help='hidden dimension',
        default=CONFIG['Parameters']['hidden_dim']
    )
    parser.add_argument(
        '--nhead', type=int,
        help='nhead',
        default=CONFIG['Parameters']['nhead']
    )
    parser.add_argument(
        '--nlayers', type=int,
        help='nlayers',
        default=CONFIG['Parameters']['nlayers']
    )
    parser.add_argument(
        '--class-num', type=int,
        help='class dimension',
        default=CONFIG['Parameters']['class_num']
    )
    parser.add_argument(
        '--max-len', type=int,
        help='sequence max len',
        default=CONFIG['Parameters']['max_len']
    )
    parser.add_argument(
        '--act', type=str,
        help='activation',
        default=CONFIG['Parameters']['activation']
    )
    parser.add_argument(
        '--one-hot-label', type=bool,
        help='if true then use one-hot vector as label input, else integer',
        default=CONFIG['Parameters']['one_hot_label']
    )

    # training param
    parser.add_argument(
        '--lr', type=float,
        help='lr',
        default=CONFIG['Parameters']['lr'])
    parser.add_argument(
        '--batch-size', type=int,
        help='batch-size',
        default=CONFIG['Parameters']['batch_size']
    )
    parser.add_argument(
        '--epoch', type=int,
        default=CONFIG['Parameters']['epochs']
    )

    # dir param
    dir = CONFIG["Save_dir"]
    parser.add_argument(
        '--save-dir', type=str,
        default=f'{dir}{t.lower()}_transformer_al.pt'
    )

    return parser.parse_args()


def dataloader(args):
    '''還不夠格式化'''

    news_train = load_dataset(CONFIG['dataset'], split='train')
    news_test = load_dataset(CONFIG['dataset'], split='test')

    # TODO: columns
    train_text = [b['text'] for b in news_train]
    train_label = multi_class_process(
        [b['label'] for b in news_train], args.class_num
    )
    test_text = [b['text'] for b in news_test]
    test_label = multi_class_process(
        [b['label'] for b in news_test], args.class_num
    )

    clean_train = [data_preprocessing(t, True) for t in train_text]
    clean_test = [data_preprocessing(t, True) for t in test_text]

    # NOTE: 這是要回傳的值
    vocab = create_vocab(clean_train)

    clean_train_id = convert2id(clean_train, vocab)
    clean_test_id = convert2id(clean_test, vocab)

    max_len = max([len(s) for s in clean_train_id])
    print('max seq length', max_len)
    args.max_len = min(max_len, args.max_len)

    train_features, train_mask = PadTransformer(clean_train_id, args.max_len)
    test_features, test_mask = PadTransformer(clean_test_id, args.max_len)

    X_train, X_valid, mask_train, mask_valid, y_train, y_valid = train_test_split(
        train_features, train_mask, train_label, test_size=0.2, random_state=1)
    X_test, mask_test, y_test = test_features, test_mask, test_label

    print('dataset information:')
    print('=====================')
    print('train size', len(X_train))
    print('valid size', len(X_valid))
    print('test size', len(test_features))
    print('=====================')

    train_data = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(mask_train),
        torch.stack(y_train)
    )
    test_data = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(mask_test),
        torch.stack(y_test)
    )
    valid_data = TensorDataset(
        torch.from_numpy(X_valid),
        torch.from_numpy(mask_valid),
        torch.stack(y_valid)
    )

    batch_size = args.batch_size

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, vocab


def load_parameters():

    global CONFIG
    with open("configs/hyperparameters.json", "r", encoding="utf8") as f:
        CONFIG = json.load(f)


def save_parameters():

    with open("configs/hyperparameters.json", "w", encoding="utf8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, sort_keys=True, indent=3)


def train(args):

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader, vocab = dataloader(args)

    if args.pretrained:
        w = get_word_vector(vocab, emb=args.pretrained)
    else:
        w = None

    act = get_act(args)
    emb = EmbeddingAL(
        (args.vocab_size, args.class_num),
        (args.emb_dim, args.label_dim),
        lin=args.one_hot_label,
        pretrained=w,
        act=act
    )

    nhead = 6
    l1 = TransformerEncoderAL((args.emb_dim, args.label_dim), nhead,
                              args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)
    l2 = TransformerEncoderAL((args.emb_dim, args.hid_dim), nhead,
                              args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)
    l3 = TransformerEncoderAL((args.emb_dim, args.hid_dim), nhead,
                              args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)
    l4 = TransformerEncoderAL((args.emb_dim, args.hid_dim), nhead,
                              args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)
    l5 = TransformerEncoderAL((args.emb_dim, args.hid_dim), nhead,
                              args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)
    l6 = TransformerEncoderAL((args.emb_dim, args.hid_dim), nhead,
                              args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act)

    model = TransformerForCLS(emb, l1, l2, l3, l4, l5, l6)
    model = model.to(device)
    print(count_parameters(model))

    trainer = TransfomerTrainer(
        model, args.lr, train_loader=train_loader,
        valid_loader=valid_loader, test_loader=test_loader,
        save_dir=args.save_dir, is_al=True
    )
    trainer.run(epochs=args.epoch)
    trainer.eval()


if __name__ == '__main__':
    args = arg_parser()
    train(args)
    save_parameters()
