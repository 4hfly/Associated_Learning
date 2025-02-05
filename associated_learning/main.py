# -*- coding: utf-8 -*-
import argparse
import json

import torch

from .datasets import AGNews, DBpedia, IMDb, SST2
from .models import EmbeddingAL, LSTMAL, TransformerEncoderAL
from .models import LSTMForCLS, LSTMALForCLS, TransformerForCLS, TransformerALForCLS
from .trainers import ALTrainer, Trainer, TransfomerTrainer
from .utils import count_parameters, get_act, get_word_vector

# Mode: (str) 'LSTM' | 'LSTMAL' | 'Transformer' | 'TransformerAL'
# Dataset: (str) 'AGNews' | 'DBpedia' | 'IMDb' | 'SST'
# Parameters:
#   `pretrained`: (str) 'fasttext' | 'glove' | None
#   `activation`: (str)
CONFIG = {
    'Title': 'Template',
    'Mode': 'template',
    'Dataset': 'template',
    'Parameters': {
        'vocab_size': 30000,
        'pretrained': 'glove',
        'embedding_dim': 300,
        'label_dim': 128,
        'hidden_dim': 512,
        'nhead': 6,
        'nlayers': 2,
        'class_num': 2,
        'max_len': 256,
        'one_hot_label': True,
        'activation': 'tanh',
        'lr': 1e-3,
        'batch_size': 256,
        'epochs': 50,
        'ramdom_label': False
    },
    "Save_dir": 'ckpt/',
}


def arg_parser():

    t = CONFIG['Title']
    m = CONFIG['Mode']
    parser = argparse.ArgumentParser(f'{t} for {m} training.')

    # model params
    parser.add_argument(
        '--vocab_size', type=int,
        help='vocab-size',
        default=CONFIG['Parameters']['vocab_size']
    )
    parser.add_argument(
        '--pretrained', type=str,
        default=CONFIG['Parameters']['pretrained']
    )
    parser.add_argument(
        '--emb_dim', type=int,
        help='word embedding dimension',
        default=CONFIG['Parameters']['embedding_dim']
    )
    parser.add_argument(
        '--hid_dim', type=int,
        help='hidden dimension',
        default=CONFIG['Parameters']['hidden_dim']
    )
    parser.add_argument(
        '--label_dim', type=int,
        help='label hidden dimension',
        default=CONFIG['Parameters']['label_dim']
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
        '--class_num', type=int,
        help='class dimension',
        default=CONFIG['Parameters']['class_num']
    )
    parser.add_argument(
        '--act', type=str,
        help='activation',
        default=CONFIG['Parameters']['activation']
    )
    parser.add_argument(
        '--one_hot_label', type=bool,
        help='if true then use one-hot vector as label input, else integer',
        default=CONFIG['Parameters']['one_hot_label']
    )

    # training params
    parser.add_argument(
        '--lr', type=float,
        help='lr',
        default=CONFIG['Parameters']['lr'])
    parser.add_argument(
        '--batch_size', type=int,
        help='batch-size',
        default=CONFIG['Parameters']['batch_size']
    )
    parser.add_argument(
        '--epoch', type=int,
        default=CONFIG['Parameters']['epochs']
    )
    parser.add_argument(
        '--max_len', type=int,
        default=CONFIG['Parameters']['max_len']
    )

    # dir param
    dir = CONFIG["Save_dir"]
    parser.add_argument(
        '--save_dir', type=str,
        default=f'{dir}{t.lower()}_transformer.pt'
    )

    return parser.parse_args()


def train(args):

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if CONFIG['Dataset'] == 'AGNews':
        data = AGNews()
    elif CONFIG['Dataset'] == 'DBpedia':
        data = DBpedia()
    elif CONFIG['Dataset'] == 'IMDb':
        data = IMDb()
    elif CONFIG['Dataset'] == 'SST':
        data = SST2()

    if CONFIG['Mode'] == 'LSTM' or CONFIG['Mode'] == 'LSTMAL':
        train_loader, valid_loader, test_loader, vocab = data.load(args)
    elif CONFIG['Mode'] == 'Transformer' or CONFIG['Mode'] == 'TransformerAL':
        train_loader, valid_loader, test_loader, vocab = data.load_with_masks(
            args)

    if args.pretrained:
        w = get_word_vector(vocab, emb=args.pretrained)
    else:
        w = None

    if CONFIG['Mode'] == 'LSTM':

        model = LSTMForCLS(
            args.vocab_size, args.emb_dim, args.hid_dim,
            args.nlayers, args.class_num, pretrained=w
        )
        model = model.to(device)
        print(count_parameters(model))

        trainer = Trainer(
            model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader,
            save_dir=args.save_dir
        )

    elif CONFIG['Mode'] == 'LSTMAL':

        act = get_act(args)
        emb = EmbeddingAL(
            (args.vocab_size, args.class_num),
            (args.emb_dim, args.label_dim),
            lin=args.one_hot_label, pretrained=w, act=act
        )
        l1 = LSTMAL(
            args.emb_dim, args.label_dim,
            (args.hid_dim, args.hid_dim), dropout=0,
            bidirectional=True, act=act
        )
        l2 = LSTMAL(
            2 * args.hid_dim, args.hid_dim,
            (args.hid_dim, args.hid_dim), dropout=0,
            bidirectional=True, act=act
        )

        model = LSTMALForCLS(emb, l1, l2)
        model = model.to(device)

        trainer = ALTrainer(
            model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader,
            save_dir=args.save_dir
        )

    elif CONFIG['Mode'] == 'Transformer':

        model = TransformerForCLS(
            args.vocab_size, args.emb_dim, args.hid_dim,
            args.nhead, args.nlayers, args.class_num, pretrained=w
        )
        model = model.to(device)
        print(count_parameters(model))

        trainer = TransfomerTrainer(
            model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader,
            save_dir=args.save_dir, is_al=False
        )

    elif CONFIG['Mode'] == 'TransformerAL':

        act = get_act(args)
        emb = EmbeddingAL(
            (args.vocab_size, args.class_num),
            (args.emb_dim, args.label_dim),
            lin=args.one_hot_label,
            pretrained=w,
            act=act
        )

        nhead = 6
        l1 = TransformerEncoderAL(
            (args.emb_dim, args.label_dim), nhead,
            args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act
        )
        l2 = TransformerEncoderAL(
            (args.emb_dim, args.hid_dim), nhead,
            args.hid_dim, args.hid_dim, dropout=0.1, batch_first=True, act=act
        )

        model = TransformerALForCLS(emb, l1, l2)
        model = model.to(device)
        print(count_parameters(model))

        trainer = TransfomerTrainer(
            model, args.lr, train_loader=train_loader,
            valid_loader=valid_loader, test_loader=test_loader,
            save_dir=args.save_dir, is_al=True
        )

    trainer.run(epochs=args.epoch)

    if CONFIG['Dataset'] == 'SST':
        trainer.pred()
    else:
        trainer.eval()

    if CONFIG['Mode'] == 'Transformer' or CONFIG['Mode'] == 'TransformerAL':
        trainer.short_cut_emb()
        trainer.short_cut_l1()


def load_parameters():

    global CONFIG
    with open("configs/hyperparameters.json", "r", encoding="utf8") as f:
        CONFIG = json.load(f)


def save_parameters():

    with open("configs/hyperparameters.json", "w", encoding="utf8") as f:
        json.dump(CONFIG, f, ensure_ascii=False, sort_keys=True, indent=3)


if __name__ == '__main__':
    load_parameters()
    args = arg_parser()
    train(args)
