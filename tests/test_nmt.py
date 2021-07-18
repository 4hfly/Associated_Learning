# -*- coding: utf-8 -*-
import sys

import torch

sys.path.append("../Associated_Learning/")
from nmt.nmt_with_org import NMT, compute_corpus_level_bleu_score, get_tkrs

args = {
    "src": "en",
    "src_tkr": "../data/tokenizer/spm/en/bpe.model",
    "tgt_tkr": "../data/tokenizer/spm/fr/bpe.model",
}

src_tkr, tgt_tkr = get_tkrs(args)

MODEL = NMT(
    embed_size=8,
    hidden_size=16,
    dropout_rate=0.,
    input_feed=1,
    label_smoothing=0.1,
    src_tkr=src_tkr,
    tgt_tkr=tgt_tkr)

MODEL.train()
# TODO: -*- ... settings ... -*-


def test_tkrs():

    encoded = src_tkr.Encode('<s> This is a test </s>')
    assert src_tkr.Decode(encoded) == '<s> This is a test </s>'
    encoded = tgt_tkr.Encode('<s> Bonjour, vous tous! Comment ça va 😁? </s>')
    assert tgt_tkr.Decode(encoded) == '<s> Bonjour, vous tous! Comment ça va  ⁇ ? </s>'


def test_decode():

    # 有需要再設計
    src_encodings = []
    src_sent_masks = []
    decoder_init_vec = ()
    tgt_sents_var = []

    # 隨便設
    ans = torch.FloatTensor([1, 2])

    output = MODEL.decode(src_encodings, src_sent_masks,
                          decoder_init_vec, tgt_sents_var)

    assert output == ans


def test_beam_search():

    src_sent = []
    beam_size = 1
    max_decoding_time_step = 10

    ans = [(["<s>", "good", "</s>"], 100.)]

    output = MODEL.beam_search(src_sent, beam_size, max_decoding_time_step)

    assert output == ans


def test_compute_corpus_level_bleu_score():

    references = ["good dog the good dog"]
    hypotheses = ["dog dog dog dog dog"]

    ans = 0.

    output = compute_corpus_level_bleu_score(references, hypotheses)

    assert output == ans


test_tkrs()
