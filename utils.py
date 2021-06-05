#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from tokenizers import decoders
from tokenizer import BPETokenizer


def al_loss():

    pass


def bpe(input, files="data/vocab_fr.json"):

    tokenizer = BPETokenizer(files=files)
    encoded = tokenizer.encode(input)
    decoded = tokenizer.decode(encoded)
    return  encoded, decoded


def test():
    encoded, decoded = bpe("Bonjour, vous tous ! Comment ça va 😁 ?")
    print(encoded.tokens)
    print(decoded)


if __name__ == "__main__":
    test()
