#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: 還有 SentencePiece 可用

from typing import Any, List, Tuple, Union

from tokenizers import Encoding, Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# https://huggingface.co/docs/tokenizers/python/latest/quicktour.html
# TODO: 1. post-processing 2. batch


class BPETokenizer(object):

    def __init__(self, vocab_size=30000, min_freq=0, files=None) -> None:
        super(BPETokenizer, self).__init__()

        if files is not None:
            self.tokenizer = Tokenizer.from_file(files)
        else:
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
        )

        self.tokenizer.pre_tokenizer = Whitespace()

    def train(self, files=None) -> None:

        if files is None:
            files = [
                f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]

        # files 長這樣：["test.txt", "train.txt", "valid.txt"]
        self.tokenizer.train(files, self.trainer)

    def save(self) -> None:

        self.tokenizer.save("data/tokenizer-wiki.json")

    def encode(self, input: Union[str, List[str], Tuple[str]]) -> Encoding:

        return self.tokenizer.encode(input)


if __name__ == "__main__":

    tokenizer = BPETokenizer()
    tokenizer.train()
    tokenizer.save()
    encoded = tokenizer.encode("Hello, y'all! How are you 😁 ?")
    print(encoded.tokens)
