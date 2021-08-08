# -*- coding: utf-8 -*-
from tokenizer import BPETokenizer


def test_bpe():

    def bpe(
        input,
        lang="fr",
        files=[
            "data/tokenizer/fr/vocab.json",
            "data/tokenizer/fr/merges.txt"
        ]
    ):

        tokenizer = BPETokenizer(lang=lang, files=files)
        encoded = tokenizer.encode(input)
        decoded = tokenizer.decode(encoded)
        return encoded, decoded

    encoded, decoded = bpe("Bonjour, vous tous! Comment ça va 😁?")
    print(encoded.tokens)
    print(decoded)