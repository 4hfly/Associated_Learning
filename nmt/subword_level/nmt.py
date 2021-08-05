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

        # TODO: EmbAL
        self.emb_x = nn.Embedding(
            len(vocab.src), embed_size, padding_idx=vocab.src['<pad>'])
        self.emb_y = nn.Embedding(
            len(vocab.src), embed_size, padding_idx=vocab.tgt['<pad>'])
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.reverse = reverse

        # TODO: LSTMAL. Default layers: 2.
        self.f1 = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.f2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.g1 = nn.LSTM(embed_size, hidden_size, bidirectional=True)
        self.g2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)
        self.hf1 = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.hf2 = nn.LSTMCell(hidden_size * 2, hidden_size)
        self.hg1 = nn.LSTMCell(embed_size + hidden_size, hidden_size)
        self.hg2 = nn.LSTMCell(hidden_size * 2, hidden_size)
        # NOTE: attention decoder params.
        # h_projection == hidden state init
        # c_projection == cell state init
        # att_projection == att_src_linear
        # combined_output_projection == att_vec_linear
        self.h_proj_x1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.h_proj_x2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.h_proj_y1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.h_proj_y2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.c_proj_x1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.c_proj_x2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.c_proj_y1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.c_proj_y2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_proj_x1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_proj_x2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_proj_y1 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.att_proj_y2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.combined_output_proj_x1 = nn.Linear(
            hidden_size * 3, hidden_size, bias=False)
        self.combined_output_proj_x2 = nn.Linear(
            hidden_size * 3, hidden_size, bias=False)
        self.combined_output_proj_y1 = nn.Linear(
            hidden_size * 3, hidden_size, bias=False)
        self.combined_output_proj_y2 = nn.Linear(
            hidden_size * 3, hidden_size, bias=False)
        # for softmax
        self.target_vocab_projection = nn.Linear(
            hidden_size, len(vocab.tgt), bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        # NOTE: general NMT form.
        # self.encoder = nn.LSTMAL(embed_size, hidden_size, bidirectional=True)
        # self.decoder = nn.LSTMCell(embed_size + hidden_size, hidden_size)
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

        # TODO: AL emb layer.
        emb_x = self.emb(source_padded)
        emb_y = self.emb(target_padded)

        # -*- Layer 1 -*-
        # TODO: AL encoder.
        packed_x = pack_padded_sequence(
            emb_x, source_lengths, enforce_sorted=False)
        packed_y = pack_padded_sequence(
            emb_y, target_lengths, enforce_sorted=False)
        encoded_x, (h_x, c_x) = self.f1(packed_x)
        encoded_y, (h_y, c_y) = self.g1(packed_y)
        encoded_x, _ = pad_packed_sequence(encoded_x)
        encoded_y, _ = pad_packed_sequence(encoded_y)

        # masks
        x_masks = self.generate_sent_masks(encoded_x, source_lengths)
        y_masks = self.generate_sent_masks(encoded_y, target_lengths)

        # TODO: AL decoder.
        if not self.reverse:
            # TODO: should be encoded_y here if using auto-encoder.
            encoded_x = encoded_x.permute(1, 0, 2)
            decoder_init_hidden_y = self.h_proj_y1(
                torch.cat((h_y[0], h_y[1]), 1))
            decoder_init_cell_y = self.c_proj_y1(
                torch.cat((c_y[0], c_y[1]), 1))
            decoder_init_state_y = (decoder_init_hidden_y, decoder_init_cell_y)

            # Init
            decoder_state = decoder_init_state_y
            combined_outputs = []
            # Initialize previous combined output vector o_{t-1} as zero
            # TODO: encoded_y
            batch_size = encoded_x.size(0)
            o_prev = torch.zeros(
                batch_size, self.hidden_size, device=self.device)
            att_proj = self.att_proj_y1(encoded_x)

            for y_t in emb_y.split(split_size=1):
                y_t = y_t.squeeze(0)
                ybar_t = torch.cat((y_t, o_prev), 1)
                decoder_state, o_t, _ = self.step(
                    ybar_t, decoder_state, encoded_x, att_proj, x_masks)
                combined_outputs.append(o_t)
                o_prev = o_t

            combined_outputs = torch.stack(combined_outputs)

        else:
            # TODO: encoded_x
            encoded_y = encoded_y.permute(1, 0, 2)
            decoder_init_hidden_x = self.h_proj_x1(
                torch.cat((h_x[0], h_x[1]), 1))
            decoder_init_cell_x = self.c_proj_x1(
                torch.cat((c_x[0], c_x[1]), 1))
            decoder_init_state_x = (decoder_init_hidden_x, decoder_init_cell_x)

            # Init
            decoder_state = decoder_init_state_x
            combined_outputs = []
            # Initialize previous combined output vector o_{t-1} as zero
            # TODO: encoded_x
            batch_size = encoded_y.size(0)
            o_prev = torch.zeros(
                batch_size, self.hidden_size, device=self.device)
            att_proj = self.att_proj_x1(encoded_y)

            for x_t in emb_y.split(split_size=1):
                x_t = x_t.squeeze(0)
                xbar_t = torch.cat((x_t, o_prev), 1)
                decoder_state, o_t, _ = self.step(
                    xbar_t, decoder_state, encoded_y, att_proj, y_masks)
                combined_outputs.append(o_t)
                o_prev = o_t

            combined_outputs = torch.stack(combined_outputs)
        # -*- End of Layer 1 -*-

        # -*- Layer 2 -*-
        # same as layer 1.
        packed_x = pack_padded_sequence(
            emb_x, source_lengths, enforce_sorted=False)
        packed_y = pack_padded_sequence(
            emb_y, target_lengths, enforce_sorted=False)
        encoded_x, (h_x, c_x) = self.f2(packed_x)
        encoded_y, (h_y, c_y) = self.g2(packed_y)
        encoded_x, _ = pad_packed_sequence(encoded_x)
        encoded_y, _ = pad_packed_sequence(encoded_y)

        # masks
        x_masks = self.generate_sent_masks(encoded_x, source_lengths)
        y_masks = self.generate_sent_masks(encoded_y, target_lengths)

        if not self.reverse:
            # TODO: encoded_y
            encoded_x = encoded_x.permute(1, 0, 2)
            decoder_init_hidden_y = self.h_proj_y2(
                torch.cat((h_y[0], h_y[1]), 1))
            decoder_init_cell_y = self.c_proj_y2(
                torch.cat((c_y[0], c_y[1]), 1))
            decoder_init_state_y = (decoder_init_hidden_y, decoder_init_cell_y)

            # Init
            decoder_state = decoder_init_state_y
            combined_outputs = []
            # Initialize previous combined output vector o_{t-1} as zero
            # TODO: encoded_y
            batch_size = encoded_x.size(0)
            o_prev = torch.zeros(
                batch_size, self.hidden_size, device=self.device)
            att_proj = self.att_proj_y2(encoded_x)

            for y_t in emb_y.split(split_size=1):
                y_t = y_t.squeeze(0)
                ybar_t = torch.cat((y_t, o_prev), 1)
                decoder_state, o_t, _ = self.step(
                    ybar_t, decoder_state, encoded_x, att_proj, x_masks)
                combined_outputs.append(o_t)
                o_prev = o_t

            combined_outputs = torch.stack(combined_outputs)

        else:
            # TODO: encoded_x
            encoded_y = encoded_y.permute(1, 0, 2)
            decoder_init_hidden_x = self.h_proj_x2(
                torch.cat((h_x[0], h_x[1]), 1))
            decoder_init_cell_x = self.c_proj_x2(
                torch.cat((c_x[0], c_x[1]), 1))
            decoder_init_state_x = (decoder_init_hidden_x, decoder_init_cell_x)

            # Init
            decoder_state = decoder_init_state_x
            combined_outputs = []
            # Initialize previous combined output vector o_{t-1} as zero
            # TODO: encoded_x
            batch_size = encoded_y.size(0)
            o_prev = torch.zeros(
                batch_size, self.hidden_size, device=self.device)
            att_proj = self.att_proj_x2(encoded_y)

            for x_t in emb_y.split(split_size=1):
                x_t = x_t.squeeze(0)
                xbar_t = torch.cat((x_t, o_prev), 1)
                decoder_state, o_t, _ = self.step(
                    xbar_t, decoder_state, encoded_y, att_proj, y_masks)
                combined_outputs.append(o_t)
                o_prev = o_t

            combined_outputs = torch.stack(combined_outputs)
        # -*- End of Layer 1 -*-

        # TODO: loss_d here.
        P = F.log_softmax(self.target_vocab_projection(
            combined_outputs), dim=-1)

        # Zero out, probabilities for which we have nothing in the target text
        target_masks = (target_padded != self.vocab.tgt['<pad>']).float()

        # Compute log probability of generating true target words
        target_gold_words_log_prob = torch.gather(
            P, index=target_padded[1:].unsqueeze(-1), dim=-1).squeeze(-1) * target_masks[1:]
        scores = target_gold_words_log_prob.sum(dim=0)

        return scores

    def step(
        self,
        Ybar_t: torch.Tensor,
        dec_state: Tuple[torch.Tensor, torch.Tensor],
        enc_hiddens: torch.Tensor,
        enc_hiddens_proj: torch.Tensor,
        enc_masks: torch.Tensor
    ) -> Tuple[Tuple, torch.Tensor, torch.Tensor]:
        """
        Compute one forward step of the LSTM decoder, including the attention computation.

        @param Ybar_t (Tensor): Concatenated Tensor of [Y_t o_prev], with shape (b, e + h). The input for the decoder,
                                where b = batch size, e = embedding size, h = hidden size.
        @param dec_state (tuple(Tensor, Tensor)): Tuple of tensors both with shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's prev hidden state, second tensor is decoder's prev cell.
        @param enc_hiddens (Tensor): Encoder hidden states Tensor, with shape (b, src_len, h * 2), where b = batch size,
                                    src_len = maximum source length, h = hidden size.
        @param enc_hiddens_proj (Tensor): Encoder hidden states Tensor, projected from (h * 2) to h. Tensor is with shape (b, src_len, h),
                                    where b = batch size, src_len = maximum source length, h = hidden size.
        @param enc_masks (Tensor): Tensor of sentence masks shape (b, src_len),
                                    where b = batch size, src_len is maximum source length. 

        @returns dec_state (tuple (Tensor, Tensor)): Tuple of tensors both shape (b, h), where b = batch size, h = hidden size.
                First tensor is decoder's new hidden state, second tensor is decoder's new cell.
        @returns combined_output (Tensor): Combined output Tensor at timestep t, shape (b, h), where b = batch size, h = hidden size.
        @returns e_t (Tensor): Tensor of shape (b, src_len). It is attention scores distribution.
                                Note: You will not use this outside of this function.
                                      We are simply returning this value so that we can sanity check
                                      your implementation.
        """
        pass

    def generate_sent_masks(self, enc_hiddens: torch.Tensor, source_lengths: List[int]) -> torch.Tensor:
        """
        Generate sentence masks for encoder hidden states.

        Args:
            enc_hiddens (Tensor):
            encodings of shape (b, src_len, 2*h), where b = batch size,
            src_len = max source length, h = hidden size.
            source_lengths (List[int]):
            List of actual lengths for each of the sentences in the batch.

        Returns:
            enc_masks (Tensor):
            Tensor of sentence masks of shape (b, src_len), where src_len = max source length, h = hidden size.
        """
        enc_masks = torch.zeros(enc_hiddens.size(
            0), enc_hiddens.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(source_lengths):
            enc_masks[e_id, src_len:] = 1
        return enc_masks.to(self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[Hypothesis]:
        """ Given a single source sentence, perform beam search, yielding translations in the target language.
        @param src_sent (List[str]): a single source sentence (words)
        @param beam_size (int): beam size
        @param max_decoding_time_step (int): maximum number of time steps to unroll the decoding RNN
        @returns hypotheses (List[Hypothesis]): a list of hypothesis, each hypothesis has two fields:
                value: List[str]: the decoded target sentence, represented as a list of words
                score: float: the log-likelihood of the target sentence
        """
        src_sents_var = self.vocab.src.to_input_tensor([src_sent], self.device)

        src_encodings, dec_init_vec = self.encode(
            src_sents_var, [len(src_sent)])
        src_encodings_att_linear = self.att_projection(src_encodings)

        h_tm1 = dec_init_vec
        att_tm1 = torch.zeros(1, self.hidden_size, device=self.device)

        eos_id = self.vocab.tgt['</s>']

        hypotheses = [['<s>']]
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

            y_tm1 = torch.tensor([self.vocab.tgt[hyp[-1]]
                                  for hyp in hypotheses], dtype=torch.long, device=self.device)
            y_t_embed = self.emb_y(y_tm1)

            x = torch.cat([y_t_embed, att_tm1], dim=-1)

            (h_t, cell_t), att_t, _ = self.step(x, h_tm1,
                                                exp_src_encodings, exp_src_encodings_att_linear, enc_masks=None)

            # log probabilities over target words
            log_p_t = F.log_softmax(
                self.target_vocab_projection(att_t), dim=-1)

            live_hyp_num = beam_size - len(completed_hypotheses)
            contiuating_hyp_scores = (hyp_scores.unsqueeze(
                1).expand_as(log_p_t) + log_p_t).view(-1)
            top_cand_hyp_scores, top_cand_hyp_pos = torch.topk(
                contiuating_hyp_scores, k=live_hyp_num)

            prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
            hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

            new_hypotheses = []
            live_hyp_ids = []
            new_hyp_scores = []

            for prev_hyp_id, hyp_word_id, cand_new_hyp_score in zip(prev_hyp_ids, hyp_word_ids, top_cand_hyp_scores):
                prev_hyp_id = prev_hyp_id.item()
                hyp_word_id = hyp_word_id.item()
                cand_new_hyp_score = cand_new_hyp_score.item()

                hyp_word = self.vocab.tgt.id2word[hyp_word_id]
                new_hyp_sent = hypotheses[prev_hyp_id] + [hyp_word]
                if hyp_word == '</s>':
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
            hyp_scores = torch.tensor(
                new_hyp_scores, dtype=torch.float, device=self.device)

        if len(completed_hypotheses) == 0:
            completed_hypotheses.append(Hypothesis(value=hypotheses[0][1:],
                                                   score=hyp_scores[0].item()))

        completed_hypotheses.sort(key=lambda hyp: hyp.score, reverse=True)

        return completed_hypotheses

    @ property
    def device(self) -> torch.device:
        # TODO: EmbAL
        return self.emb_x.weight.device

    @ staticmethod
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
