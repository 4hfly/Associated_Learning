#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tokenizer import BPETokenizer


def al_loss():

    pass


def bpe(input, files="data/tokenizer-wiki.json"):

    tokenizer = BPETokenizer(files=files)
    return tokenizer.encode(input)


def test():
    encoded = bpe("Hello, y'all! How are you 😁 ?")
    print(encoded.tokens)


if __name__ == "__main__":
    test()
