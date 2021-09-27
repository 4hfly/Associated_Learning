import argparse
import json
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from transformer.encoder import TransformerEncoder
from transformer.encoder.utils import PositionalEncoding
from utils import *

os.environ["WANDB_SILENT"] = "true"

# TODO: 可以寫成一個 class 取代 parser。
# Parameters:
#   `pretrained`: str | None，不建議用 'none'
#   `activation`: str
CONFIG = {
    'Title': 'SST2',
    'dataset': 'sst2',
    'Parameters': {
        'vocab_size': 30000,
        'pretrained': 'glove',
        'embedding_dim': 300,
        'hidden_dim': 256,
        'nhead': 6,
        'nlayers': 2,
        'class_num': 2,
        'max_len': 256,
        'one_hot_label': True,
        'activation': 'tanh',
        'lr': 1e-3,
        'batch_size': 256,
        'epochs': 5,
        'ramdom_labe;': False
    },
    "Save_dir": 'data/ckpt/',
}


class TransformerForCLS(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        nhead,
        nlayers,
        class_num,
        dropout: float = 0.5,
        pretrain: Tensor = None
    ):

        super(TransformerForCLS, self).__init__()

        self.emb_dim = embedding_dim

        if pretrain == None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(
                pretrain, freeze=False, padding_idx=0
            )

        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            embedding_dim, nhead, hidden_dim, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        # NOTE: (default) batch first = True
        # self.encoder = TransformerEncoder(
        #     embedding_dim, hidden_dim, nhead, nlayers, dropout
        # )
        # NOTE: transformer decoder
        # decoder_layer = nn.TransformerDecoderLayer(
        #     embedding_dim, nhead, hidden_dim, dropout=dropout, batch_first=True
        # )
        # self.decoder = nn.TransformerDecoder(decoder_layer, nlayers)
        self.fc = nn.Linear(embedding_dim, class_num)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):

        x = self.embedding(x)
        # NOTE: positional encoding.
        # 乘上 math.sqrt(self.emb_dim) 是模仿別人的，可以改。
        x = x * math.sqrt(self.emb_dim)
        x = self.pos_encoder(x)

        # NOTE: 假如是 transformer package 的，forward 參數只會有兩個，因此寫法是
        # output = self.encoder(x, src_key_padding_mask).sum(dim=1)
        # nn.Transformer 有三個參數，而 encoder 不需要 n*n 的 mask。
        output = self.encoder(x, src_mask, src_key_padding_mask)

        # NOTE: 9.18
        # device = output.device
        # tgt_mask = self._generate_square_subsequent_mask(
        #     output.size(1)).to(device)
        # output = self.decoder(
        #     x, output, tgt_mask=None, memory_mask=None,
        #     tgt_key_padding_mask=src_key_padding_mask,
        #     memory_key_padding_mask=src_key_padding_mask
        # )
        output = output.sum(dim=1)

        src_len = (src_key_padding_mask == 0).sum(dim=1)
        # fit the shape of output
        src_len = torch.stack((src_len,) * output.size(1), dim=1)
        output = output / src_len
        output = self.fc(output)

        return self.logsoftmax(output)

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
        default=f'{dir}{t.lower()}_transformer.pt'
    )

    return parser.parse_args()


def dataloader(args):
    '''還不夠格式化'''

    dataset = CONFIG['dataset']
    train_df = pd.read_csv(f'data/{dataset}/train.tsv', sep='\t')
    valid_df = pd.read_csv(f'data/{dataset}/dev.tsv', sep='\t')
    test_df = pd.read_csv(f'data/{dataset}/test.tsv', sep='\t')

    # TODO: columns
    train_text = train_df['sentence'].tolist()
    train_label = multi_class_process(train_df['label'].tolist(), 2)

    valid_text = valid_df['sentence'].tolist()
    valid_label = multi_class_process(valid_df['label'].tolist(), 2)

    test_text = test_df['sentence'].tolist()

    clean_train = [data_preprocessing(t) for t in train_text]
    clean_valid = [data_preprocessing(t) for t in valid_text]
    clean_test = [data_preprocessing(t) for t in test_text]

    # NOTE: 這是要回傳的值
    vocab = create_vocab(clean_train)

    clean_train_id = convert2id(clean_train, vocab)
    clean_valid_id = convert2id(clean_valid, vocab)
    clean_test_id = convert2id(clean_test, vocab)

    cti = []
    tl = []
    for i in range(len(clean_train_id)):
        if len(clean_train_id[i]) >= 1:
            cti.append(clean_train_id[i])
            tl.append(train_label[i])
    clean_train_id = cti
    train_label = tl

    max_len = max([len(s) for s in clean_train_id])
    print('max seq length', max_len)

    train_features, train_mask = PadTransformer(clean_train_id, max_len)
    valid_features, valid_mask = PadTransformer(clean_valid_id, max_len)
    test_features, test_mask = PadTransformer(clean_test_id, max_len)

    X_train, mask_train, y_train = train_features, train_mask, train_label
    X_valid, mask_valid, y_valid = valid_features, valid_mask, valid_label
    X_test, mask_test = test_features, test_mask

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
    )
    valid_data = TensorDataset(
        torch.from_numpy(X_valid),
        torch.from_numpy(mask_valid),
        torch.stack(y_valid)
    )

    batch_size = args.batch_size

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, vocab


def load_parameters():

    global CONFIG
    t = CONFIG['Title']
    with open(f'configs/{t}/hyperparameters.json', 'r', encoding='utf8') as f:
        CONFIG = json.load(f)


def save_parameters():

    with open('configs/hyperparameters.json', 'w', encoding='utf8') as f:
        json.dump(CONFIG, f, ensure_ascii=False, sort_keys=True, indent=3)


def train(args):

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, valid_loader, test_loader, vocab = dataloader(args)

    if args.pretrained:
        w = get_word_vector(vocab, emb=args.pretrained)
    else:
        w = None

    model = TransformerForCLS(
        args.vocab_size, args.emb_dim, args.hid_dim,
        args.nhead, args.nlayers, args.class_num, pretrain=w
    )
    model = model.to(device)
    print(count_parameters(model))

    trainer = TransfomerTrainer(
        model, args.lr, train_loader=train_loader,
        valid_loader=valid_loader, test_loader=test_loader,
        save_dir=args.save_dir, is_al=False
    )
    trainer.run(epochs=args.epoch)
    trainer.pred()


if __name__ == '__main__':
    args = arg_parser()
    train(args)
    # save_parameters()
