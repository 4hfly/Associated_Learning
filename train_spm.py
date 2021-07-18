# -*- coding: utf-8 -*-
import os
import sentencepiece as spm

dir = ['data/tokenizer/spm/en', 'data/tokenizer/spm/fr']
for d in dir:
    if not os.path.exists(d):
        os.makedirs(d)

# en
spm.SentencePieceTrainer.train(
    f"--input=data/wmt14/commoncrawl/commoncrawl.fr-en.en,\
data/wmt14/europarl_v7/europarl-v7.fr-en.en,\
data/wmt14/giga/giga-fren.release2.fixed.en,\
data/wmt14/news-commentary/news-commentary-v9.fr-en.en,\
data/wmt14/un/undoc.2000.fr-en.en\
    --model_prefix=data/tokenizer/spm/en/bpe\
    --user_defined_symbols=<s>,</s>\
    --vocab_size=25000\
    --model_type=bpe"
)

# fr
spm.SentencePieceTrainer.train(
    f"--input=data/wmt14/commoncrawl/commoncrawl.fr-en.fr,\
data/wmt14/europarl_v7/europarl-v7.fr-en.fr,\
data/wmt14/giga/giga-fren.release2.fixed.fr,\
data/wmt14/news-commentary/news-commentary-v9.fr-en.fr,\
data/wmt14/un/undoc.2000.fr-en.fr\
    --model_prefix=data/tokenizer/spm/fr/bpe\
    --user_defined_symbols=<s>,</s>\
    --vocab_size=25000\
    --model_type=bpe"
)