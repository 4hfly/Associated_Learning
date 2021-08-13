# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
# from torchtext.data.functional import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
from torchtext.vocab import FastText

from torchtext.vocab import Vocab

from model import LSTMAL, EmbeddingAL, CLS
import torch.nn.functional as F
from collections import Counter

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class Dataset(object):

    def __init__(self, train_data):
        
        self.train_data = train_data
        self.create_vocab()
        self.train_data = train_data
        self.vocab_size = self.vocab.__len__
        self.dataloader = DataLoader(
            train_data, batch_size=8, collate_fn=self.collate_fn)

    def create_vocab(self):

        tokenizer = get_tokenizer('basic_english')
        counter = Counter()
        with tqdm(unit_scale=0, unit='lines') as t:
            for tokens in self.yield_tokens(self.train_data, tokenizer):
                counter.update(tokens)
                t.update(1)

        self.vocab = Vocab(counter, min_freq=10)
        self.text_pipeline = lambda x: self.vocab(tokenizer(x))
        self.label_pipeline = lambda x: 1 if x == 'pos' else 0

    @staticmethod
    def yield_tokens(iter, tokenizer):

        for _, text in iter:
            yield tokenizer(text)

    def collate_fn(self, batch):

        label_list, text_list, offsets = [], [], [0]

        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(
                self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))

        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)

        return label_list.to(self.device), text_list.to(self.device), offsets.to(self.device)


from datasets import Dataset
from model import LSTMAL, EmbeddingAL


class Trainer(object):

    def __init__(self, dataset) -> None:

        self.dataloader = dataset.load()
        # self.dataloader = dataset.dataloader
        print(list(self.dataloader))
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # TODO: emb_size for y
        # TODO: magic number (300, 2)
        pretrained = FastText()
        self.embedding = EmbeddingAL(
            (dataset.vocab_size, 2), (300, 128), pretrained)
        self.layer_1 = LSTMAL(300, 128, (128, 128))
        self.layer_2 = LSTMAL(128, 128, (64, 64))
        self.model = nn.Sequential(
            self.embedding,
            self.layer_1,
            self.layer_2
        )
        print('AL parameter num', get_n_params(self.model))
        self.optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer_2 = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer_3 = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.model = CLS(dataset.vocab_size, 300, 400)
        print('parameter num',get_n_params(self.model))
        self.optimizer_1 = torch.optim.Adam(self.model.parameters(), lf=1e-4)

    def train(self, epoch):

        self.model.train()

        # log params
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text) in enumerate(self.dataloader):

            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            self.optimizer_3.zero_grad()

            out = self.model(text)
            loss = F.cross_entropy_loss(out, label)
            loss.backward()
            self.optimizer_1.step()
            predicted_label = out
            '''
            # emb layer
            x, y = self.embedding(text, label)
            loss = self.embedding.loss()
            loss.backward()
            self.optimizer_1.step()

            # 1st lstm
            x, hx, y = self.layer_1(x, y)
            loss = self.layer_1.loss()
            loss.backward()
            self.optimizer_2.step()

            # 2nd lstm
            x, hx, y = self.layer_2(x, y, hx)
            loss = self.layer_2.loss()
            loss.backward()
            self.optimizer_3.step()

            # inference
            left = self.layer_2.f(self.layer_1.f(self.embedding.f(text)))
            right = self.layer_2.bx(left)
            predicted_label = self.embedding.dy(
                self.layer_1.dy(self.layer_2.dy(right)))
            '''

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    f'| epoch {epoch:3d} | {idx:5d}/{len(self.dataloader):5d} batches | accuracy {total_acc/total_count:8.3f}')
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(self):

        self.model.eval()

        total_acc, total_count = 0, 0

        with torch.no_grad():
            for label, text in self.dataloader:
                out = self.model(text)
                predicted_label = out
                '''
                left = self.layer_2.f(self.layer_1.f(self.embedding(text)))
                right = self.layer_2.bx(left)
                predicted_label = self.embedding.dy(
                    self.layer_1.dy(self.layer_2.dy(right)))
                '''
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)

        return total_acc/total_count


def main():

    train_data = IMDB(split='train')
    test_data = IMDB(split='test')
    dataset = Dataset(train_data)
    trainer = Trainer(dataset)

    for epoch in range(1, 11):
        epoch_start_time = time.time()
        trainer.train(epoch)


if __name__ == '__main__':
    main()
