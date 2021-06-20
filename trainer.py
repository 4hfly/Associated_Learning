#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import torch
import torch.nn as nn


from dataset import CoolDataset
from model import CNN, Model, MLP, ResNet, VGG


class Trainer(object):

    def __init__(self) -> None:

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = Model().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)


def main():

    pass
