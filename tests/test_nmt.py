# -*- coding: utf-8 -*-

import torch

from nmt.nmt_with_org import NMT, get_tkrs, compute_corpus_level_bleu_score

args = {
    "src": "en",
    "src_tkr": "test_data/sample/en/",
    "tgt_tkr": "test_data/sample/fr/",
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
