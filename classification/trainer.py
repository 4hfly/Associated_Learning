# -*- coding: utf-8 -*-
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchtext.vocab import FastText

from model import CLS, LSTMAL, EmbeddingAL
from read_raw import IMDB, get_emb


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


class Trainer(object):

    def __init__(self, dataset, testset) -> None:

        self.dataloader = DataLoader(
            dataset, batch_size=4, collate_fn=dataset.collate, shuffle=True)
        self.testloader = DataLoader(
            testset, batch_size=4, collate_fn=testset.collate)

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        # TODO: emb_size for y
        # TODO: magic number (300, 2)
        # NOTE: hid_size_1 = (128, 128), hid_size_2 = (hid_size_1[0] * 2, hid_size_1[1])
        emb = FastText()
        self.embedding = EmbeddingAL((
            30000, 2), (300, 128), pretrained=emb)
        self.layer_1 = LSTMAL(300, 128, (128, 128),
                              batch_first=True, bidirectional=True)
        self.layer_2 = LSTMAL(
            256, 128, (128, 128), batch_first=True, bidirectional=True)
        self.layers = nn.Sequential(
            self.embedding,
            self.layer_1,
            self.layer_2
        )
        self.layers = self.layers.to(self.device)
        print('AL parameter num', get_n_params(self.layers))

        emb = get_emb(dataset.vocab)
        self.model = CLS(30000, 325, 400, emb=emb)
        self.model = self.model.to(self.device)
        print('parameter num', get_n_params(self.model))

        self.optimizer_1 = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer_2 = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer_3 = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train(self, epoch):

        # NOTE:
        self.layers.train()

        # log params
        total_acc, total_count = 0, 0
        log_interval = 20
        start_time = time.time()
        idx = 0
        total_loss = 0
        total_batch = 0

        for i, (text, label) in enumerate(self.dataloader):

            idx += 1
            total_batch += 1
            self.optimizer_1.zero_grad()
            self.optimizer_2.zero_grad()
            self.optimizer_3.zero_grad()
            text, label = text.to(self.device), label.to(self.device)

            # emb layer
            x, y = self.embedding(text, label)
            loss = self.embedding.loss()
            loss.backward()
            total_loss += loss.item()
            self.optimizer_1.step()

            # 1st lstm
            x, hx, y = self.layer_1(x, y)
            loss = self.layer_1.loss()
            loss.backward()
            total_loss += loss.item()
            self.optimizer_2.step()

            # 2nd lstm
            x, hx, y = self.layer_2(x, y, hx)
            loss = self.layer_2.loss()
            loss.backward()
            total_loss += loss.item()
            self.optimizer_3.step()

            # inference
            x = self.embedding.f(text)
            x, (h, c) = self.layer_1.f(x)
            x, (h, c) = self.layer_2.f(x, (h, c))
            h = h.view(1, -1, 256)
            y = self.layer_2.bx(h)
            y = self.layer_2.dy(y)
            y = self.layer_1.dy(y)
            predicted_label = torch.round(self.embedding.dy(y).squeeze())

            total_acc += (predicted_label == label.to(torch.float)).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(
                    f'| epoch {epoch:3d} | {idx:5d}/{len(self.dataloader):5d} batches | accuracy {total_acc/total_count:8.3f} | loss {total_loss/total_batch:8.3f}')
                total_acc, total_count = 0, 0
                total_loss, total_batch = 0, 0
                start_time = time.time()

    def evaluate(self):

        # NOTE:
        self.layers.eval()

        total_acc, total_count = 0, 0

        with torch.no_grad():

            for text, label in self.testloader:
                text, label = text.to(self.device), label.to(self.device)

                x = self.embedding.f(text)
                x, (h, c) = self.layer_1.f(x)
                x, (h, c) = self.layer_2.f(x, (h, c))
                h = h.view(1, -1, 256)
                y = self.layer_2.bx(h)
                y = self.layer_2.dy(y)
                y = self.layer_1.dy(y)
                predicted_label = torch.round(self.embedding.dy(y).squeeze())

            total_acc += (predicted_label == label.to(torch.float)).sum().item()
            total_count += label.size(0)

        return total_acc/total_count


def main():

    train = IMDB('train')
    test = IMDB('test')

    trainer = Trainer(train, test)
    best_val_acc = -1
    for epoch in range(1, 30):
        epoch_start_time = time.time()
        trainer.train(epoch)
        val_acc = trainer.evaluate()
        print('epoch', epoch, 'valid acc', val_acc)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(trainer.model.state_dict(), './lstm_imdb.pth')
    # trainer.model.load_state_dict(torch.load('./lstm_imdb.pth'))
    # best_acc = trainer.eval()
    print('this is the best acc', best_val_acc)


if __name__ == '__main__':
    main()
