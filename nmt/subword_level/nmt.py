# -*- coding: utf-8 -*-
import math
import pickle
import sys
import time
from collections import namedtuple

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
# TODO: same as SmoothingFunction.
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import sentencepiece as spm

import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model import EmbAL, LSTMAL
from utils import read_corpus, batch_iter, LabelSmoothingLoss, to_input_tensor

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2, reverse=False):
        super(NMT, self).__init__()

        self.emb = EmbAL(embed_size, vocab)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.reverse = reverse

        self.layer = LSTMAL(embed_size, hidden_size, bidirectional=True)
        # NOTE: general NMT form.
        # self.encoder = nn.LSTMAL(embed_size, hidden_size, bidirectional=True)
        # self.decoder = nn.LSTMCell(embed_size, hidden_size)
        # self.h_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # self.c_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # self.att_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        # self.combined_output_projection = nn.Linear(
        #     hidden_size * 3, hidden_size, bias=False)
        # self.target_vocab_projection = nn.Linear(
        #     hidden_size, len(vocab.tgt), bias=False)
        # self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, source: List[List[str]], target: List[List[str]]) -> torch.Tensor:
        """
        take a mini-batch of source and target sentences, compute the log-likelihood of
        target sentences.
        Args:
            src_sents: list of source sentence tokens
            tgt_sents: list of target sentence tokens, wrapped by `<s>` and `</s>`
        Returns:
            scores: a variable/tensor of shape (batch_size, ) representing the
                log-likelihood of generating the gold-standard target sentence for
                each example in the input batch
        """

        # Compute sentence lengths
        source_lengths = [len(s) for s in source]
        target_lengths = [len(s) for s in target]

        # Convert list of lists into tensors
        # TODO: `to_input_tensor()` is in vocab.py. We should wrap the target sentence
        # in token <s> and token </s> before calling this function.
        source_padded = self.vocab.src.to_input_tensor(
            source, device=self.device)   # Tensor: (src_len, b)
        target_padded = self.vocab.tgt.to_input_tensor(
            target, device=self.device)   # Tensor: (tgt_len, b)

        # Run the network forward:
        # 1. Apply the encoder to `source_padded` by calling `self.encode()`
        # 2. Generate sentence masks for `source_padded` by calling `self.generate_sent_masks()`
        # 3. Apply the decoder to compute combined-output by calling `self.decode()`
        # 4. Compute log probability distribution over the target vocabulary using the
        # combined_outputs returned by the `self.decode()` function.

        # NOTE: deprecated codes.
        # enc_hiddens, dec_init_state = self.encode(
        #     source_padded, source_lengths)
        # enc_masks = self.generate_sent_masks(enc_hiddens, source_lengths)
        # combined_outputs = self.decode(
        #     enc_hiddens, enc_masks, dec_init_state, target_padded)
        emb_x = self.emb(source_padded)
        emb_y = self.emb(target_padded)

        # TODO: AL decoder.
        x, (h_x, c_x), y, (h_y, c_y) = self.layer(emb_x, emb_y)
        combined_outputs = self.layer.inference()

        P = F.log_softmax(self.target_vocab_projection(
            combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(
            P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)

        return scores

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """
        Given a single source sentence, perform beam search
        Args:
            src_sent: a single tokenized source sentence
            beam_size: beam size
            max_decoding_time_step: maximum number of time steps to unroll the decoding RNN
        Returns:
            hypotheses: a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """

        src_sents_var, src_sents_len = self.to_input_tensor(
            [src_sent], self.src_tkr, self.device)
        src_encodings, dec_init_vec = self.encode(src_sents_var, src_sents_len)
        src_encodings_att_linear = self.att_src_linear(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.tgt_tkr.token_to_id('</s>')
        sos_id = self.tgt_tkr.token_to_id('<s>')

        hypotheses = [[sos_id]]
        hyp_scores = torch.zeros(
            len(hypotheses), dtype=torch.float, device=self.device)
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < max_decoding_time_step:
            t += 1
            hyp_num = len(hypotheses)

            exp_src_encodings = src_encodings.expand(hyp_num,
                                                     src_encodings.size(1),
                                                     src_encodings.size(2))

            exp_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                           src_encodings_att_linear.size(
                                                                               1),
                                                                           src_encodings_att_linear.size(2))

            # y_tm1 = torch.tensor([self.tgt_tkr.token_to_id(hyp[-1]) for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_tm1 = torch.tensor(
                [hyp[-1] for hyp in hypotheses], dtype=torch.long, device=self.device)

            y_tm1_embed = self.tgt_embed(y_tm1)

            if self.input_feed:
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1,
                                                      exp_src_encodings, exp_src_encodings_att_linear, src_sent_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(self.readout(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(
                1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos // 25000
            hyp_word_ids = top_cand_hyp_pos % 25000

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()
                # hyp_word = self.tgt_tkr.decode([hyp_word_id])
                hyp_word = hyp_word_id
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == eos_id:
                    completed_hypotheses.append(Hypothesis(value=new_hyp_sent[1:-1],
                                                           score=cand_new_hyp_score))
                else:
                    new_hypotheses.append(new_hyp_sent)
                    live_hyp_ids.append(prev_hyp_id)
                    new_hyp_scores.append(cand_new_hyp_score)

            if len(completed_hypotheses) == beam_size:
                break

            live_hyp_ids = torch.tensor(
                live_hyp_ids, dtype=torch.long, device=self.device)
            h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
            att_tm1 = att_t[live_hyp_ids]

            hypotheses = new_hypotheses
            # raise Exception('ok')
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)
        # print(completed_hypotheses)
        # raise Exception('ok')
        return completed_hypotheses

    @property
    def device(self) -> torch.device:
        return self.src_embed.weight.device

    @staticmethod
    def load(model_path: str):
        params = torch.load(
            model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(src_tkr=params['src_tkr'],
                    tgt_tkr=params['tgt_tkr'], **args)
        model.load_state_dict(params['state_dict'])

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.embed_size, hidden_size=self.hidden_size, dropout_rate=self.dropout_rate,
                         input_feed=self.input_feed, label_smoothing=self.label_smoothing),
            'src_tkr': self.src_tkr,
            'tgt_tkr': self.tgt_tkr,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)
