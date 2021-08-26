#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchvision import datasets

from datasets import CoolDataset
from models import CNN, MLP, VGG, AutoEncoder, ResNet
from utils import al_loss


class Trainer(object):

    def __init__(self) -> None:

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ResNet().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

def main():

    pass
