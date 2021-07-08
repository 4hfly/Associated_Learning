import json
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from utils import LabelSmoothingLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class EmbAL(nn.Module):
    def __init__(self, emb_dim, vocab_size=25000):
        self.f = nn.Embedding(vocab_size, emb_dim)
        self.g = nn.Embedding(vocab_size, emb_dim)
        self.decoder_f = nn.Linear(emb_dim, vocab_size)
        self.decoder_g = nn.Linear(emb_dim, vocab_size)
        self.labelsmoothingloss = LabelSmoothingLoss(0.1, 25000)

    def forward(self, x, y, reverse=False):
        '''
        input params:
            x: (src_len, batch_size)
            y: (tgt_len, batch_size)
        out params:
            emb_x: (src_len, batch_size, emb_dim)
            emb_y = (tgt_len, batch_size, emb_dim)
        '''

        emb_x = self.f(x)
        emb_y = self.g(y)
        loss_b = self.mse_loss(emb_x, emb_y, reverse) # bridge loss
        if not reverse:
            loss_d = self.decode_loss(emb_y, y, reverse)
        else:
            loss_d = self.decode_loss(emb_x, x, reverse)
        self.loss = loss_b + loss_d
        return emb_x.detach(), emb_y.detach()

    def loss(self):
        return self.loss

    def inference(self, x, tgt, reverse=False):

        '''
        input params:
            x: (tgt_len-1, batch_size, hidden)
            tgt: (tgt_len-1, batch_size)
        output params:
            out: (tgt_len-1, batch_size, vocab_size)
        '''

        if not reverse:
            out = F.logsoftmax(self.decoder_g(x))
        else:
            out = F.logsoftmax(self.decoder_f(x))
        return out

    def mse_loss(self, x, y):

        '''
        input params:
            x: (src_len, batch_size, emb_dim)
            y: (tgt_len, batch_size, emb_dim)
        '''
        x = x[x.nonzero(as_tuple=True)].view(x.size(0), x.size(1), -1).mean(1)
        y = y[y.nonzero(as_tuple=True)].view(y.size(0), y.size(1), -1).mean(1)
        return F.mse_loss(x, y)
    
    def decode_loss(self, pred, tgt, reverse=False):

        '''
        input params:
            pred: (tgt_len -1, batch_size, hidden)
            tgt: (tgt_len - 1, batch_size)
        parameters:
            word_prob: (tgt_len - 1, batch_size, vocab_size)

        '''

        if not reverse:
            word_prob = F.logsoftmax(self.decoder_g(pred)) # readout layer 
        else:
            word_prob = F.logsoftmax(self.decoder_f(pred)) # readout layer
        tgt_words_to_pred = torch.count_nonzero(tgt)
        prob = -self.labelsmoothingloss(word_prob.reshape(-1, word_prob.size(-1)), tgt[1:].view(-1)).view(-1, pred.size(1)).sum(0)
        prob = prob.sum() / pred.size(1)
        return prob



class LSTMAL(nn.Module):
    def __init__(self, input_dim, hid_dim, bidirectional, dropout, input_feed, reverse):

        self.hid_dim = hid_dim
        self.input_feed = input_feed

        self.f = nn.LSTM(input_dim, hid_dim, bidirectional=bidirectional)
        self.g = nn.LSTM(input_dim, hid_dim, bidirectional=bidirectional)

        self.decoder_f_cell_init = nn.Linear(self.hid_dim*2, self.hid_dim)
        self.decoder_g_cell_init = nn.Linear(self.hid_dim*2, self.hid_dim)

        decoder_lstm_input = input_dim + hid_dim if self.input_feed else input_dim

        self.decoder_f = nn.LSTMCell(decoder_lstm_input, hid_dim)
        self.deocder_g = nn.LSTMCell(decoder_lstm_input, hid_dim)

        self.att_src_f_linear = nn.Linear(self.hid_dim * 2, self.hid_dim, bias=False)
        self.att_src_g_linear = nn.Linear(self.hid_dim * 2, self.hid_dim, bias=False)
        self.att_vec_f_linear = nn.Linear(self.hid_dim * 2 + self.hid_dim, self.hid_dim, bias=False)
        self.att_vec_g_linear = nn.Linear(self.hid_dim * 2 + self.hid_dim, self.hid_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.reverse = reverse


    def forward(self, x, y, src_sent_lens, tgt_sent_lens, tgt_emb, tgt_sents, reverse=False):
        '''
        input params:
            x: (src_len, batch_size, emb_dim)
            y: (tgt_len, batch_size, emb_dim)
        '''
        packed_x = pack_padded_sequence(x, src_sent_lens, enforce_sorted=False)
        packed_y = pack_padded_sequence(y, tgt_sent_lens, enforce_sorted=False)

        out_x, (h_x, c_x) = self.f(packed_x)
        out_y, (h_y, c_y) = self.g(packed_y)

        out_x, _ = pad_packed_sequence(out_x)
        out_y, _ = pad_packed_sequence(out_y)

        loss_b = self.mse_loss(h_x, h_y)

        if not reverse: # en 2 fr

            src_encodings = out_y.permute(1, 0, 2)
            dec_init_cell = self.decoder_g_cell_init(torch.cat([c_y[0], c_y[1]], dim=1))
            dec_init_state = torch.tanh(dec_init_cell)    
            src_sent_masks = self.get_attention_mask(src_encodings, src_sent_lens)
            attn_vecs = self.decode(out_x, src_sent_masks, (dec_init_state, dec_init_cell), tgt_sents, tgt_emb, True)    
            loss_d = self.decode_loss(attn_vecs, x)

        else: # fr 2 en

            src_encodings = out_x.permute(1, 0, 2)
            dec_init_cell = self.decoder_f_cell_init(torch.cat([c_x[0], c_x[1]], dim=1))
            dec_init_state = torch.tanh(dec_init_cell)    
            src_sent_masks = self.get_attention_mask(src_encodings, src_sent_lens)
            attn_vecs = self.decode(out_x, src_sent_masks, (dec_init_state, dec_init_cell), tgt_sents, tgt_emb)
            loss_d = self.decode_loss(attn_vecs, y)

        self.loss = loss_b + loss_d

    def get_attention_mask(self, src_encodings: torch.Tensor, src_sents_len: List[int]) -> torch.Tensor:
        src_sent_masks = torch.zeros(src_encodings.size(0), src_encodings.size(1), dtype=torch.float)
        for e_id, src_len in enumerate(src_sents_len):
            src_sent_masks[e_id, src_len:] = 1

        return src_sent_masks.to(self.device)

    def decode(self, src_encodings, src_sent_masks, decoder_init_vec, tgt_sents_var, tgt_emb, reverse=False):

        # (batch_size, src_sent_len, hidden_size)
        src_encoding_att_linear = self.att_src_linear(src_encodings)

        batch_size = src_encodings.size(0)

        # initialize the attentional vector
        att_tm1 = torch.zeros(batch_size, self.hid_dim, device=self.device)

        # (tgt_sent_len, batch_size, embed_size)
        # here we omit the last word, which is always </s>.
        # Note that the embedding of </s> is not used in decoding
        tgt_word_embeds = tgt_emb(tgt_sents_var) # tgt_emb is a layer

        h_tm1 = decoder_init_vec

        att_ves = []

        # start from y_0=`<s>`, iterate until y_{T-1}
        for y_tm1_embed in tgt_word_embeds.split(split_size=1):
            y_tm1_embed = y_tm1_embed.squeeze(0)
            if self.input_feed:
                # input feeding: concate y_tm1 and previous attentional vector
                # (batch_size, hidden_size + embed_size)
                x = torch.cat([y_tm1_embed, att_tm1], dim=-1)
            else:
                x = y_tm1_embed

            (h_t, cell_t), att_t, alpha_t = self.step(x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks)

            att_tm1 = att_t
            h_tm1 = h_t, cell_t
            att_ves.append(att_t)

        # (tgt_sent_len - 1, batch_size, tgt_vocab_size)
        att_ves = torch.stack(att_ves)

        return att_ves

    def step(self, x, h_tm1, src_encodings, src_encoding_att_linear, src_sent_masks, reverse=False):
        # h_t: (batch_size, hidden_size)
        if not reverse:
            h_t, cell_t = self.decoder_g(x, h_tm1)
        else:
            h_t, cell_t = self.decoder_f(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encoding_att_linear, src_sent_masks)
        if not reverse:
            att_t = torch.tanh(self.att_vec_g_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        else:
            att_t = torch.tanh(self.att_vec_f_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)      
        att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t

    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask):
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)

        if mask is not None:
            att_weight.data.masked_fill_(mask.bool(), -float('inf'))

        softmaxed_att_weight = F.softmax(att_weight, dim=-1)

        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(softmaxed_att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, softmaxed_att_weight
        
    def decode_loss(self, x, y):
        x = x.reshape(-1, x.size(-1))
        y = y.reshape(-1, y.size(-1))
        return F.mse_loss(x,y)

    def mse_loss(self, x, y):
        x = x.reshape(-1, self.hid_dim)
        y = y.reshape(-1, self.hid_dim)
        return F.mse_loss(x, y)

    def inference(self, emb_x, src_sent_lens, tgt_emb, tgt_sents_var, reverse):
        if not reverse:
            packed_x = pack_padded_sequence(emb_x, src_sent_lens, enforce_sorted=False)
            out_x, (h_x, c_x) = self.f(packed_x)
            out_x, _ = pad_packed_sequence(out_x)
            
            src_encodings = out_x.permute(1, 0, 2)
            dec_init_cell = self.decoder_g_cell_init(torch.cat([c_x[0], c_x[1]], dim=1))
            dec_init_state = torch.tanh(dec_init_cell)    
            src_sent_masks = self.get_attention_mask(src_encodings, src_sent_lens)
            attn_vecs = self.decode(out_x, src_sent_masks, (dec_init_state, dec_init_cell), tgt_sents_var, tgt_emb, True)    

        else:
            packed_x = pack_padded_sequence(emb_x, src_sent_lens, enforce_sorted=False)
            out_x, (h_x, c_x) = self.g(packed_x)
            out_x, _ = pad_packed_sequence(out_x)
            
            src_encodings = out_x.permute(1, 0, 2)
            dec_init_cell = self.decoder_f_cell_init(torch.cat([c_x[0], c_x[1]], dim=1))
            dec_init_state = torch.tanh(dec_init_cell)    
            src_sent_masks = self.get_attention_mask(src_encodings, src_sent_lens)
            attn_vecs = self.decode(out_x, src_sent_masks, (dec_init_state, dec_init_cell), tgt_sents_var, tgt_emb, True)    
        return attn_vecs