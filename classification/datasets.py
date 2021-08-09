from collections import Counter
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = get_tokenizer('basic_english')
vocab: Vocab


class Dataset(object):

    def __init__(self, train_data) -> None:

        self.train_data = train_data
        self.vocab = create_vocab(train_data)
        self.vocab_size = len(self.vocab.itos)

    def load(self):
        return DataLoader(self.train_data, batch_size=8, collate_fn=collate_fn)


def create_vocab(train_data):

    global vocab

    counter = Counter()
    for tokens in yield_tokens(train_data, tokenizer):
        counter.update(tokens)

    vocab = Vocab(counter, vectors='fasttext.en.300d')
    return vocab


def yield_tokens(iter, tokenizer):

    for _, text in iter:
        yield tokenizer(text)


def text_pipeline(x):
    return vocab(tokenizer(x))


def label_pipeline(x):
    return 1 if x == 'pos' else 0


def collate_fn(batch):

    label_list, text_list = [], []

    for _label, _text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(
            text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(torch.cat(text_list), padding_value=1.)

    return label_list.to(device), text_list.to(device)
