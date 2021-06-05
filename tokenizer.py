#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: 還有 SentencePiece 可用

from typing import Any, List, Tuple, Union

from tokenizers import Encoding, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Lowercase, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
# TODO: 1. post-processing 2. batch


class BPETokenizer(object):

    def __init__(
        self,
        vocab_size=25000,
        min_freq=5,
        lang="en",
        files=None
    ) -> None:
        super(BPETokenizer, self).__init__()

        if files is not None:
            self.tokenizer = Tokenizer.from_file(files)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        self.lang = lang
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["[UNK]", "[SEP]", "[PAD]"],
            initial_alphabet=ByteLevel.alphabet()
        )

        self.tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.decoder = ByteLevelDecoder()

    def train(self, files=None) -> None:

        if files is None:
            files = [
                f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]

        # files 長這樣：["test.txt", "train.txt", "valid.txt"]
        self.tokenizer.train(files, self.trainer)

    def save(self) -> None:

        self.tokenizer.save(f"data/vocab_{self.lang}.json")

    def encode(self, input: Union[str, List[str], Tuple[str]]) -> Encoding:

        return self.tokenizer.encode(input)

    def decode(self, input: Encoding) -> str:

        return self.tokenizer.decode(input.ids)


if __name__ == "__main__":

    tokenizer = BPETokenizer(lang="fr")
    files = [
        "data/wmt14/commoncrawl/commoncrawl.fr-en.fr",
        "data/wmt14/europarl_v7/europarl-v7.fr-en.fr",
        "data/wmt14/giga/giga-fren.release2.fixed.fr",
        "data/wmt14/news-commentary/news-commentary-v9.fr-en.fr",
        "data/wmt14/un/undoc.2000.fr-en.fr"
        ]
    tokenizer.train()
    tokenizer.save()
    encoded = tokenizer.encode("Bonjour, vous tous ! Comment ça va 😁 ?")
    print(encoded.tokens)
    decoded = tokenizer.decode(encoded)
    print(decoded)
