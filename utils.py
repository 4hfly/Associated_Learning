#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tokenizer import ByteLevelBPETokenizer

DATA = [
    "data/wmt14/commoncrawl/commoncrawl.fr-en.fr",
    "data/wmt14/europarl_v7/europarl-v7.fr-en.fr",
    "data/wmt14/giga/giga-fren.release2.fixed.fr",
    "data/wmt14/news-commentary/news-commentary-v9.fr-en.fr",
    "data/wmt14/un/undoc.2000.fr-en.fr"
]


def preprocess_text():

    for file in DATA:
        with open(file, "r", encoding="utf8") as f:
            data = f.readlines()

        with open(f"{file}.shell", "w", encoding="utf8") as f:
            for s in data:
                f.write(f"<s> {s.strip()} </s>\n")


def test():

    def bpe(
        input,
        lang="fr",
        files=[
            "data/tokenizer/fr/vocab.json",
            "data/tokenizer/fr/merges.txt"
        ]
    ):

        tokenizer = ByteLevelBPETokenizer(lang=lang, files=files)
        encoded = tokenizer.encode(input)
        decoded = tokenizer.decode(encoded)
        return encoded, decoded

    encoded, decoded = bpe("Bonjour, vous tous! Comment ça va 😁?")
    print(encoded.tokens)
    print(decoded)


if __name__ == "__main__":
    test()
