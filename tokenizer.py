#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: 還有 SentencePiece 可用

from typing import List, Tuple, Union

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
        files=[None, None]
    ) -> None:
        """

        Args:
            vocab_size: (int)
            min_freq: minimum frequency
            lang: 
            files: (List[str]) ["vocab.json", "merge.txt"]
        """
        super(BPETokenizer, self).__init__()

        self.tokenizer = Tokenizer(BPE(files[0], files[1]))

        self.lang = lang
        self.trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=["[PAD]", "[SEP]"],
            initial_alphabet=ByteLevel.alphabet()
        )

        # https://huggingface.co/docs/tokenizers/python/latest/components.html#normalizers
        self.tokenizer.normalizer = Sequence([NFKC(), Lowercase()])
        # https://huggingface.co/docs/tokenizers/python/latest/components.html#pre-tokenizers
        self.tokenizer.pre_tokenizer = ByteLevel()
        self.tokenizer.decoder = ByteLevelDecoder()

    def train(self, files=None) -> None:

        if files is None:
            # files 長這樣：["test.txt", "train.txt", "valid.txt"]
            files = [
                f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]
            ]

        self.tokenizer.train(files, self.trainer)

    def save(self) -> None:

        self.tokenizer.model.save(f"data/tokenizer/{self.lang}")

    def encode(self, input: Union[str, List[str], Tuple[str]]) -> Encoding:

        return self.tokenizer.encode(input)

    def decode(self, input: Encoding) -> str:

        # 注意 type(input) == Encoding
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
    # Outputs:
    # ['Ġbon', 'j', 'our', ',', 'Ġv', 'ous', 'Ġto', 'us', 'Ġ!', 'Ġcomment',
    #  'ĠÃ', '§', 'a', 'Ġva', 'Ġ', 'ð', 'Ł', 'ĺ', 'ģ', 'Ġ?']
    print(encoded.tokens)
    decoded = tokenizer.decode(encoded)
    # Outputs:
    # bonjour, vous tous ! comment ça va 😁 ?
    print(decoded)
