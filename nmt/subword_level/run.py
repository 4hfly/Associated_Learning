# -*- coding: utf-8 -*-
import sys
import time

import math
import sys
import pickle
import time

from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
# import sacrebleu

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm

import torch
import torch.nn.utils

from nmt import Hypothesis, NMT
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry


def evaluate_ppl(model, dev_data, tkr, batch_size=32):
    """
    Evaluate perplexity on dev sentences
    Args:
        dev_data: a list of dev sentences
        batch_size: batch size
    Returns:
        ppl: the perplexity on dev sentences
    """

    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # you may want to wrap the following code using a context manager provided
    # by the NN library to signal the backend to not to keep gradient information
    # e.g., `torch.no_grad()`

    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(
                [len(tkr.encode(s).ids)-1 for s in tgt_sents])
            # tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score
    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference
    Returns:
        bleu_score: corpus-level BLEU score
    """

    print(references[1:5])
    print(hypotheses[1:5])

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    # for i in range(len(references)):
    #     references[i] = [x.split() for x in references[i]]
    # for i in range(len(hypotheses)):
    #     hypotheses[i] = [x.split() for x in hypotheses[i]]

    print(references[0])
    print(hypotheses[0])
    print(len(references))
    print(len(hypotheses))
    bleu_score = 0
    for i in range(len(references)):
        bleu_score += sentence_bleu(references[i], hypotheses[i])
    print(bleu_score)
    bleu_score = bleu_score*100 / len(references)
    print(bleu_score)
    # bleu_score = corpus_bleu([[ref] for ref in references],
    #                          [[hyp] for hyp in hypotheses])
    # print(bleu_score)
    # list of list of token(sent bleu), list of list of list of token(corpus bleu)
    raise Exception('ok')
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [[hyp] for hyp in hypotheses])

    return bleu_score


def get_tkrs(args):

    # TODO: replace the path with args[]
    model_file = 'data/tokenizer/spm/en/bpe.model'
    sp_src = spm.SentencePieceProcessor(model_file=model_file)
    model_file = 'data/tokenizer/spm/fr/bpe.model'
    sp_tgt = spm.SentencePieceProcessor(model_file=model_file)

    return sp_src, sp_tgt

    # mark as deprecated?
    """
    src_tkr = Tokenizer(BPE())
    src_tkr.normalizer = Sequence([
        NFKC(),
        Lowercase()
    ])
    src_tkr.pre_tokenizer = ByteLevel()
    src_tkr.decoder = ByteLevelDecoder()
    src_tkr.model = BPE(args["src_tkr"]+'/vocab.json',
                        args["src_tkr"]+'/merges.txt')

    tgt_tkr = Tokenizer(BPE())
    tgt_tkr.normalizer = Sequence([
        NFKC(),
        Lowercase()
    ])
    tgt_tkr.pre_tokenizer = ByteLevel()
    tgt_tkr.decoder = ByteLevelDecoder()
    tgt_tkr.model = BPE(args["tgt_tkr"]+'/vocab.json',
                        args["tgt_tkr"]+'/merges.txt')
    return src_tkr, tgt_tkr
    """


def check_empty(src, tgt):
    new_src = []
    new_tgt = []
    delete_num = 0
    for i in range(len(src)):
        if len(src[i]) == 0:
            delete_num += 1
            continue
        elif len(tgt[i]) == 2:
            delete_num += 1
            continue
        else:
            new_src.append(src[i])
            new_tgt.append(tgt[i])
    print('delete num', delete_num)
    return new_src, new_tgt


def train(args: Dict):
    train_data_src = read_corpus(args['train_src'], lang='src')
    train_data_tgt = read_corpus(args['train_tgt'], lang='tgt')

    train_data_src, train_data_tgt = check_empty(
        train_data_src, train_data_tgt)

    dev_data_src = read_corpus(args['dev_src'], lang='src')
    dev_data_tgt = read_corpus(args['dev_tgt'], lang='tgt')

    dev_data_src, dev_data_tgt = check_empty(dev_data_src, dev_data_tgt)

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['batch_size'])
    clip_grad = float(args['clip_grad'])
    valid_niter = int(args['valid_niter'])
    log_every = int(args['log_every'])
    model_save_path = args['save_to']

    src_tkr, tgt_tkr = get_tkrs(args)

    model = NMT(embed_size=int(args['embed_size']),
                hidden_size=int(args['hidden_size']),
                dropout_rate=float(args['dropout']),
                input_feed=args['input_feed'],
                label_smoothing=float(args['label_smoothing']),
                src_tkr=src_tkr,
                tgt_tkr=tgt_tkr)
    model.train()

    uniform_init = float(args['uniform_init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' %
              (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)

    vocab_mask = torch.ones(25000)
    vocab_mask[0] = 0

    device = torch.device("cuda:0" if args['cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1
        torch.cuda.empty_cache()

        for src_sents, tgt_sents in batch_iter(train_data, src_tkr, tgt_tkr, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            optimizer.zero_grad()

            batch_size = len(src_sents)
            # print(src_sents, tgt_sents)
            # src_lens = [len(s) for s in src_sents]
            # tgt_lens = [len(s) for s in tgt_sents]
            # if 2 in tgt_lens or 0 in src_lens:
            #     print()

            # (batch_size)
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size

            loss.backward()

            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm(
                model.parameters(), clip_grad)

            optimizer.step()

            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val

            tgt_words_num_to_predict = sum(
                [len(tgt_tkr.encode(s).ids)-1 for s in tgt_sents])

            # tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f '
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(
                                                                                             report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words /
                                                                                         (time.time(
                                                                                         ) - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cum_examples,
                                                                                             np.exp(
                                                                                                 cum_loss / cum_tgt_words),
                                                                                             cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                # dev batch size can be a bit larger
                dev_ppl = evaluate_ppl(model, dev_data, tgt_tkr, batch_size=32)
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' %
                      (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(
                    hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print(
                        'save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(),
                               model_save_path + '.optim')
                elif patience < int(args['patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['max_num_trial']):
                            print('early stop!', file=sys.stderr)
                            x = [i for i in range(len(hist_valid_scores))]
                            y = [-h for h in hist_valid_scores]
                            plt.plot(x, y)
                            src = args['src']
                            plt.savefig(f'nmt-src-{src}.png')
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * \
                            float(args['lr_decay'])
                        print(
                            'load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(
                            model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers',
                              file=sys.stderr)
                        optimizer.load_state_dict(
                            torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

                if epoch == int(args['max_epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    x = [i for i in range(len(hist_valid_scores))]
                    y = [-h for h in hist_valid_scores]
                    plt.plot(x, y)
                    src = args['src']
                    plt.savefig(f'nmt-src-{src}.png')
                    exit(0)


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:
    was_training = model.training
    model.eval()

    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(
                src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)

            hypotheses.append(example_hyps)

    if was_training:
        model.train(was_training)

    return hypotheses


def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """

    print(
        f"load test source sentences from [{args['test_src']}]", file=sys.stderr)
    test_data_src = read_corpus(args['test_src'], lang='src')
    if args['test_tgt']:
        print(
            f"load test target sentences from [{args['test_tgt']}]", file=sys.stderr)
        test_data_tgt = read_corpus(args['test_tgt'], lang='tgt')

    print(f"load model from {args['save_to']}", file=sys.stderr)
    model = NMT.load(args['save_to'])

    if args['cuda']:
        model = model.to(torch.device("cuda:0"))

    src_tkr, tgt_tkr = get_tkrs(args)

    test_data_tgt = [tgt_tkr.encode(t).ids for t in test_data_tgt]
    test_data_tgt = [tgt_tkr.decode(t) for t in test_data_tgt]

    hypotheses = beam_search(model, test_data_src,
                             beam_size=int(args['beam_size']),
                             max_decoding_time_step=int(args['max_decoding_time_step']))
    # perform target sentence decode

    if args['test_tgt']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        top_hypotheses = [tgt_tkr.decode(t.value).split()
                          for t in top_hypotheses]
        test_data_tgt = [t.split() for t in test_data_tgt]
        print(top_hypotheses[0])
        print(test_data_tgt[0])
        bleu_score = compute_corpus_level_bleu_score(
            test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['test_pred_file'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp)
            f.write(hyp_sent + '\n')


def main():
    import json
    with open(sys.argv[1]) as f:
        args = json.load(f)

    # seed the random number generators
    seed = int(args['seed'])
    torch.manual_seed(seed)
    if args['cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    if args['decode']:
        decode(args)


if __name__ == '__main__':
    main()
