import itertools
import re
import string
from collections import Counter
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torchnlp
import wandb
from nltk.corpus import stopwords
from torchnlp.word_to_vector import FastText, GloVe

from mi_tool import MI_Vis

import wandb

from vis import tsne

stop_words = set(stopwords.words('english'))


def get_act(args):
    if args.act == 'tanh':
        act = nn.Tanh()
    elif args.act == 'sigmoid':
        act = nn.Sigmoid()
    elif args.act == 'relu':
        act = nn.ReLU()
    elif args.act == 'relu6':
        act = nn.ReLU6()
    elif args.act == 'leakyrelu':
        act = nn.LeakyReLU()
    elif args.act == 'elu':
        act = nn.ELU()
    elif args.act == 'gelu':
        act = nn.GELU()
    else:
        act = None
    return act


def get_word_vector(vocab, emb='glove'):
    # vocab is a dictionary {word: word_id}
    w = []
    if emb == 'glove':
        vector = GloVe(name='6B')
    elif emb == 'FastText' or emb == 'fasttext':
        vector = FastText()

    for word in vocab.keys():
        try:
            w.append(vector[word])
        except:
            w.append(torch.zeros(300))

    return torch.stack(w, dim=0)


def get_n_params(model):

    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def acc(pred, label):
    pred = pred.argmax(-1)
    return torch.sum(pred == label.squeeze()).item()


def data_preprocessing(text, remove_stopword=False):

    text = text.lower()
    text = re.sub('<.*?>', '', text)
    text = ''.join([c for c in text if c not in string.punctuation])
    if remove_stopword:
        text = [word for word in text.split() if word not in stop_words]
    else:
        text = [word for word in text.split()]
    text = ' '.join(text)
    return text


def create_vocab(corpus, vocab_size=30000):

    corpus = [t.split() for t in corpus]
    corpus = list(itertools.chain.from_iterable(corpus))
    count_words = Counter(corpus)
    print('total count words', len(count_words))
    sorted_words = count_words.most_common()

    if vocab_size > len(sorted_words):
        v = len(sorted_words)
    else:
        v = vocab_size - 2

    vocab_to_int = {w: i + 2 for i, (w, c) in enumerate(sorted_words[:v])}

    vocab_to_int['<pad>'] = 0
    vocab_to_int['<unk>'] = 1
    print('vocab size', len(vocab_to_int))

    return vocab_to_int


def multi_class_process(labels, label_num):
    '''
    this function will convert multi-label lists into one-hot vector or n-hot vector
    '''
    hot_vecs = []
    for l in labels:
        b = torch.zeros(label_num)
        b[l] = 1
        hot_vecs.append(b)
    return hot_vecs


def multi_doubleclass_process(labels, label_num):
    '''
    this function will convert multi-label lists into one-hot vector or n-hot vector
    '''
    hot_vecs = []
    for l in labels:
        b = torch.zeros(label_num*2)
        b[l] = 1
        b[l+label_num] = 1
        hot_vecs.append(b)
    return hot_vecs


def multi_label_process(labels, label_num):
    '''
    multi-label用的
    this function will convert multi-label lists into one-hot vector or n-hot vector
    '''
    hot_vecs = []
    for l in labels:
        b = torch.zeros(label_num)
        for i in l:
            b[i] = 1
        hot_vecs.append(b)
    return hot_vecs


def convert2id(corpus, vocab_to_int):
    '''
    a list of string
    '''
    reviews_int = []
    for text in corpus:
        r = []
        for word in text.split():
            if word in vocab_to_int.keys():
                r.append(vocab_to_int[word])
            else:
                r.append(vocab_to_int['<unk>'])
        reviews_int.append(r)
    return reviews_int


def Padding(review_int, seq_len):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return features


def PadTransformer(review_int, seq_len):
    '''
    Return features of review_ints, where each review is padded with 0's or truncated to the input seq_length.
    '''
    masks = np.zeros((len(review_int), seq_len), dtype=bool)
    features = np.zeros((len(review_int), seq_len), dtype=int)
    for i, review in enumerate(review_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
            p1 = np.ones(seq_len-len(review), dtype=bool)
            p2 = np.zeros(len(review), dtype=bool)
            new_mask = np.concatenate((p1, p2), axis=0)
        else:
            new = review[: seq_len]
            new_mask = np.zeros(seq_len, dtype=bool)
        features[i, :] = np.array(new)
        masks[i, :] = new_mask
    return features, masks


class TransfomerTrainer:

    def __init__(
        self, model, lr, train_loader, valid_loader, test_loader, save_dir, label_num=None, double=False, class_num=None, is_al=False
    ):

        # Still need a arg parser to pass arguments
        # TODO: 我覺得這種外部控制的參數，用全域 CONFIG 之類的定義就好，
        # 傳參反而有點麻煩，而且 trace 會比較困難。

        self.model = model
        # TODO: wandb env.
        # project_name = save_dir.replace('/', '-')
        # wandb.init(project=project_name, entity='hibb')
        # config = wandb.config
        # wandb.watch(self.model)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.cri = nn.CrossEntropyLoss()
        self.clip = 5

        self.label_num = label_num
        self.epoch_tr_loss, self.epoch_vl_loss = [], []
        self.epoch_tr_acc, self.epoch_vl_acc = [], []
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.double = double
        self.class_num = class_num
        self.is_al = is_al

        if self.double:
            assert type(self.class_num) == int  # this is for double label

        is_cuda = torch.cuda.is_available()
        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

    def run(self, epoch):

        self.valid_acc_min = -999
        for epoch in range(epoch):

            total_loss = []
            total_emb_loss = []
            total_l1_loss = []
            total_l2_loss = []

            total_acc = 0.0
            total_count = 0

            for inputs, masks, labels in self.train_loader:

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                self.model.train()
                self.opt.zero_grad()

                if self.is_al:

                    emb_loss, l1_loss, l2_loss = self.model(inputs, labels)

                    # wandb.log({"emb loss": emb_loss.item()})
                    # wandb.log({"lstm1 loss": l1_loss.item()})
                    # wandb.log({"lstm2 loss": l2_loss.item()})
                    # wandb.log({"emb bridge loss": self.model.embedding.loss_b})
                    # wandb.log({"emb decode loss": self.model.embedding.loss_d})
                    # wandb.log({"lstm1 bridge loss": self.model.layer_1.loss_b})
                    # wandb.log({"lstm1 decode loss": self.model.layer_1.loss_d})
                    # wandb.log({"lstm2 bridge loss": self.model.layer_2.loss_b})
                    # wandb.log({"lstm2 decode loss": self.model.layer_2.loss_d})

                    loss = emb_loss + l1_loss + l2_loss

                else:

                    outputs = self.model(inputs, key_padding_mask=masks)
                    loss = self.cri(outputs, torch.argmax(labels.long(), dim=1))

                # wandb.log({"total loss": loss.item()})
                loss.backward()

                if self.is_al:
                    nn.utils.clip_grad_norm_(
                        self.model.layer_1.parameters(), self.clip, error_if_nonfinite=True
                    )
                    nn.utils.clip_grad_norm_(
                        self.model.layer_1.parameters(), self.clip, error_if_nonfinite=True
                    )
                    total_emb_loss.append(emb_loss.item())
                    total_l1_loss.append(l1_loss.item())
                    total_l2_loss.append(l2_loss.item())
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.clip, error_if_nonfinite=True
                    )

                total_loss.append(loss.item())

                self.opt.step()
                torch.cuda.empty_cache()

                # calculate the loss and perform backprop
                with torch.no_grad():

                    self.model.eval()

                    if self.is_al:

                        # TODO: mean pooling.
                        left = self.model.embedding.f(inputs)
                        output, hidden = self.model.layer_1.f(left)
                        left = self.model.layer_2.f(output, hidden)
                        left = left[:, -1, :]

                        right = self.model.layer_2.bx(left)
                        right = self.model.layer_2.dy(right)
                        right = self.model.layer_1.dy(right)
                        if self.label_num == 2:
                            predicted_label = torch.round(
                                self.model.embedding.dy(right).squeeze())
                            total_acc += (predicted_label ==
                                          labels.to(torch.float)).sum().item()
                        else:
                            if self.double is False:
                                predicted_label = self.model.embedding.dy(
                                    right)
                            else:
                                out = self.model.embedding.dy(right)
                                predicted_label = out[:, :self.class_num] + \
                                    out[:, self.class_num:]
                                labels = labels[:, :self.class_num] + \
                                    labels[:, self.class_num:]

                    else:
                        predicted_label = outputs

                    total_acc += (predicted_label.argmax(-1) ==
                                  labels.to(torch.float).argmax(-1)).sum().item()
                    total_count += labels.size(0)

            val_acc = 0.0
            val_count = 0

            for inputs, masks, labels in self.valid_loader:

                with torch.no_grad():

                    inputs = inputs.to(self.device)
                    masks = masks.to(self.device)
                    labels = labels.to(self.device)

                    self.model.eval()

                    if self.is_al:

                        left = self.model.embedding.f(inputs)
                        left = self.model.layer_1.f(left)
                        left = self.model.layer_2.f(left)
                        left = left[:, -1, :]

                        right = self.model.layer_2.bx(left)
                        right = self.model.layer_2.dy(right)
                        right = self.model.layer_1.dy(right)

                        if self.label_num == 2:
                            predicted_label = torch.round(
                                self.model.embedding.dy(right).squeeze())
                            val_acc += (predicted_label ==
                                        labels.to(torch.float)).sum().item()
                        else:
                            if self.double is False:
                                predicted_label = self.model.embedding.dy(
                                    right)
                            else:
                                out = self.model.embedding.dy(right)
                                predicted_label = out[:, :self.class_num] + \
                                    out[:, self.class_num:]
                                labels = labels[:, :self.class_num] + \
                                    labels[:, self.class_num:]

                    else:

                        predicted_label = self.model(
                            inputs, key_padding_mask=masks
                        )

                    val_acc += (predicted_label.argmax(-1) ==
                                labels.to(torch.float).argmax(-1)).sum().item()
                    val_count += labels.size(0)

            epoch_train_loss = [np.mean(total_emb_loss), np.mean(
                total_l1_loss), np.mean(total_l2_loss)]
            epoch_train_acc = total_acc/total_count
            epoch_val_acc = val_acc/val_count

            print(f'Epoch {epoch+1}')
            print(
                f'train_loss : emb loss {epoch_train_loss[0]}, layer1 loss {epoch_train_loss[1]}, layer2 loss {epoch_train_loss[2]}')
            print(
                f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
            if epoch_val_acc >= self.valid_acc_min:
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
            print(25*'==')
        final_dp = self.save_dir[:-3] + 'last.pth'
        torch.save(self.model.state_dict(), f'{final_dp}')
        print('best val acc', self.valid_acc_min)

    def eval(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, masks, labels in self.test_loader:

            with torch.no_grad():

                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)

                if self.is_al:

                    # TODO: mean pooling.
                    left = model.embedding.f(inputs)
                    left = model.layer_1.f(left)
                    left = model.layer_2.f(left)
                    left = left[:, -1, :]
                    right = model.layer_2.bx(left)
                    right = model.layer_2.dy(right)
                    right = model.layer_1.dy(right)

                    if self.label_num == 2:
                        predicted_label = torch.round(
                            model.embedding.dy(right).squeeze())
                        test_acc += (predicted_label ==
                                     labels.to(torch.float)).sum().item()
                    else:
                        if self.double is False:
                            predicted_label = self.model.embedding.dy(right)
                        else:
                            out = self.model.embedding.dy(right)
                            predicted_label = out[:, :self.class_num] + \
                                out[:, self.class_num:]
                            labels = labels[:, :self.class_num] + \
                                labels[:, self.class_num:]

                else:

                    predicted_label = self.model(
                        inputs, key_padding_mask=masks
                    )

                test_acc += (predicted_label.argmax(-1) ==
                             labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Test acc', test_acc/test_count)

    def short_cut_emb(self):

        self.model.eval()
        # self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                right = model.short_cut_emb(inputs)

                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    predicted_label = right.squeeze()
                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut emb Test acc', test_acc/test_count)

    def short_cut_l1(self):

        self.model.eval()
        # self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                right = model.short_cut_lstm(inputs)

                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    predicted_label = right.squeeze()
                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut lstm Test acc', test_acc/test_count)

    def tsne(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)
        all_feats = []
        all_labels = []
        class_num = 0
        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                left = model.embedding.f(inputs)
                output, hidden = model.layer_1.f(left)
                left, (output, c) = model.layer_2.f(output, hidden)
                left = left[:, -1, :]
                class_num = labels.size(-1)
                # left = left.reshape(left.size(1), -1)
                right = model.layer_2.bx(left)
                for i in range(right.size(0)):
                    all_feats.append(right[i, :].numpy())
                    all_labels.append(labels[i, :].argmax(-1).item())
        tsne_plot_dir = self.save_dir[:-2]+'al.tsne.png'
        tsne(all_feats, all_labels, class_num, tsne_plot_dir)
        print('tsne saved in ', tsne_plot_dir)


class ALTrainer:

    def __init__(
        self, model, lr, train_loader, valid_loader, test_loader, save_dir, label_num=None, double=False, class_num=None
    ):

        # Still need a arg parser to pass arguments
        # TODO: 我覺得這種外部控制的參數，用全域 CONFIG 之類的定義就好，
        # 傳參反而有點麻煩，而且 trace 會比較困難。

        self.mi = MI_Vis()
        self.sample = []

        self.model = model
        project_name = save_dir.replace('/', '-')
        wandb.init(project=project_name, entity='hibb')
        config = wandb.config
        wandb.watch(self.model)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.label_num = label_num
        self.epoch_tr_loss, self.epoch_vl_loss = [], []
        self.epoch_tr_acc, self.epoch_vl_acc = [], []
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        self.double = double
        self.class_num = class_num

        if self.double:
            assert type(self.class_num) == int  # this is for double label

        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

    def run(self, epoch):

        # TODO: ALTrainer 和 Trainer 這兩個的 epoch 一個有 s，一個沒有，應該要統一一下。
        self.valid_acc_min = -999
        for epoch in range(epoch):
            train_losses = []

            total_emb_loss = []
            total_l1_loss = []
            total_l2_loss = []

            total_acc = 0.0
            total_count = 0
            self.model.embedding.train()
            self.model.layer_1.train()
            self.model.layer_2.train()
            # TODO: model.train() 就會把 training 設為 True 了

            # initialize hidden state
            for inputs, labels in self.train_loader:

                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()

                emb_loss, l1_loss, l2_loss = self.model(inputs, labels)

                wandb.log({"emb loss": emb_loss.item()})
                wandb.log({"lstm1 loss": l1_loss.item()})
                wandb.log({"lstm2 loss": l2_loss.item()})
                wandb.log({"emb bridge loss": self.model.embedding.loss_b})
                wandb.log({"emb decode loss": self.model.embedding.loss_d})
                wandb.log({"lstm1 bridge loss": self.model.layer_1.loss_b})
                wandb.log({"lstm1 decode loss": self.model.layer_1.loss_d})
                wandb.log({"lstm2 bridge loss": self.model.layer_2.loss_b})
                wandb.log({"lstm2 decode loss": self.model.layer_2.loss_d})

                loss = emb_loss + l1_loss + l2_loss
                wandb.log({"total loss": loss.item()})
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.layer_1.parameters(), 5, error_if_nonfinite=True)
                nn.utils.clip_grad_norm_(
                    self.model.layer_2.parameters(), 5, error_if_nonfinite=True)

                total_emb_loss.append(emb_loss.item())
                total_l1_loss.append(l1_loss.item())
                total_l2_loss.append(l2_loss.item())

                self.opt.step()

                torch.cuda.empty_cache()

                # calculate the loss and perform backprop
                with torch.no_grad():

                    self.model.eval()

                    left = self.model.embedding.f(inputs)
                    output, hidden = self.model.layer_1.f(left)
                    left, (output, c) = self.model.layer_2.f(output, hidden)
                    left = left[:, -1, :]

                    right = self.model.layer_2.bx(left)
                    right = self.model.layer_2.dy(right)
                    right = self.model.layer_1.dy(right)
                    if self.label_num == 2:
                        predicted_label = torch.round(
                            self.model.embedding.dy(right).squeeze())
                        total_acc += (predicted_label ==
                                      labels.to(torch.float)).sum().item()
                    else:
                        if self.double is False:
                            predicted_label = self.model.embedding.dy(right)
                        else:
                            out = self.model.embedding.dy(right)
                            predicted_label = out[:, :self.class_num] + \
                                out[:, self.class_num:]
                            labels = labels[:, :self.class_num] + \
                                labels[:, self.class_num:]
                        # print(predicted_label)
                        # print(labels)
                        # raise Exception
                        total_acc += (predicted_label.argmax(-1) ==
                                      labels.to(torch.float).argmax(-1)).sum().item()

                    total_count += labels.size(0)

            # TODO: 這個 val_losses 沒用到。
            val_losses = []
            val_acc = 0.0
            val_count = 0

            self.model.embedding.eval()
            self.model.layer_1.eval()
            self.model.layer_2.eval()

            for inputs, labels in self.valid_loader:

                with torch.no_grad():

                    self.model.embedding.eval()
                    self.model.layer_1.eval()
                    self.model.layer_2.eval()

                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)

                    left = self.model.embedding.f(inputs)
                    output, hidden = self.model.layer_1.f(left)

                    left, (output, c) = self.model.layer_2.f(output, hidden)
                    left = left[:, -1, :]

                    right = self.model.layer_2.bx(left)
                    right = self.model.layer_2.dy(right)
                    right = self.model.layer_1.dy(right)
                    if self.label_num == 2:
                        predicted_label = torch.round(
                            self.model.embedding.dy(right).squeeze())
                        val_acc += (predicted_label ==
                                    labels.to(torch.float)).sum().item()
                    else:
                        if self.double is False:
                            predicted_label = self.model.embedding.dy(right)
                        else:
                            out = self.model.embedding.dy(right)
                            predicted_label = out[:, :self.class_num] + \
                                out[:, self.class_num:]
                            labels = labels[:, :self.class_num] + \
                                labels[:, self.class_num:]

                        val_acc += (predicted_label.argmax(-1) ==
                                    labels.to(torch.float).argmax(-1)).sum().item()

                    val_count += labels.size(0)

            epoch_train_loss = [np.mean(total_emb_loss), np.mean(
                total_l1_loss), np.mean(total_l2_loss)]
            epoch_train_acc = total_acc/total_count
            epoch_val_acc = val_acc/val_count

            print(f'Epoch {epoch+1}')
            print(
                f'train_loss : emb loss {epoch_train_loss[0]}, layer1 loss {epoch_train_loss[1]}, layer2 loss {epoch_train_loss[2]}')
            print(
                f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
            wandb.log({"train acc":epoch_train_acc})
            wandb.log({"valid acc":epoch_val_acc})

            if epoch_val_acc >= self.valid_acc_min:
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
            print(25*'==')
        final_dp = self.save_dir[:-3] + 'last.pth'
        torch.save(self.model.state_dict(), f'{final_dp}')
        print('best val acc', self.valid_acc_min)

    def eval(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                left = model.embedding.f(inputs)
                output, hidden = model.layer_1.f(left)
                left, (output, c) = model.layer_2.f(output, hidden)
                left = left[:, -1, :]
                # left = left.reshape(left.size(1), -1)
                right = model.layer_2.bx(left)
                right = model.layer_2.dy(right)
                right = model.layer_1.dy(right)
                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    if self.double is False:
                        predicted_label = self.model.embedding.dy(right)
                    else:
                        out = self.model.embedding.dy(right)
                        predicted_label = out[:, :self.class_num] + \
                            out[:, self.class_num:]
                        labels = labels[:, :self.class_num] + \
                            labels[:, self.class_num:]

                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)
        x = test_acc/test_count
        wandb.log({"test acc": x})
        print('Test acc', test_acc/test_count)

    def short_cut_emb(self):

        self.model.eval()
        # self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                right = model.short_cut_emb(inputs)

                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    predicted_label = right.squeeze()
                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut emb Test acc', test_acc/test_count)

    def short_cut_l1(self):

        self.model.eval()
        # self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)

        test_acc = 0
        test_count = 0

        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                right = model.short_cut_lstm(inputs)

                if self.label_num == 2:
                    predicted_label = torch.round(
                        model.embedding.dy(right).squeeze())
                    test_acc += (predicted_label ==
                                 labels.to(torch.float)).sum().item()
                else:
                    predicted_label = right.squeeze()
                    test_acc += (predicted_label.argmax(-1) ==
                                 labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)

        print('Short cut lstm Test acc', test_acc/test_count)

    def tsne_(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)
        all_feats = []
        all_labels = []
        class_num = 0
        for inputs, labels in self.test_loader:

            with torch.no_grad():

                model.embedding.eval()
                model.layer_1.eval()
                model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)
        
                left = model.embedding(inputs)
                output, hidden = model.lstm(left)
                left = output[:, -1, :]
                class_num = labels.size(-1)
                # left = left.reshape(left.size(1), -1)
                for i in range(left.size(0)):
                    all_feats.append(left[i,:].numpy())
                    all_labels.append(labels[i,:].argmax(-1).item())

                right = model.layer_2.bx(left)
                for i in range(right.size(0)):
                    all_feats.append(right[i, :].numpy())
                    all_labels.append(labels[i, :].argmax(-1).item())
        tsne_plot_dir = self.save_dir[:-2]+'al.tsne.png'
        tsne(all_feats, all_labels, class_num, tsne_plot_dir)
        print('tsne saved in ', tsne_plot_dir)


class Trainer:

    def __init__(
        self, model, lr, train_loader, valid_loader, test_loader, save_dir, label_num=None, loss_w=None
    ):

        # Still need a arg parser to pass arguments
        # TODO: 我覺得這種外部控制的參數，用全域 CONFIG 之類的定義就好，
        # 傳參反而有點麻煩，而且 trace 會比較困難。

        self.model = model
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.label_num = label_num
        self.epoch_tr_loss, self.epoch_vl_loss = [], []
        self.epoch_tr_acc, self.epoch_vl_acc = [], []
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = save_dir
        if loss_w is not None:
            self.cri = nn.CrossEntropyLoss(weight=loss_w.cuda())
        else:
            self.cri = nn.CrossEntropyLoss()
        self.clip = 5
        is_cuda = torch.cuda.is_available()

        if is_cuda:
            self.device = torch.device("cuda")
            print("GPU is available")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, CPU used")

    def run(self, epochs):

        self.valid_acc_min = -999
        # train for some number of epochs
        epoch_tr_loss, epoch_vl_loss = [], []
        epoch_tr_acc, epoch_vl_acc = [], []

        for epoch in range(epochs):
            train_losses = []
            train_acc = 0.0
            self.model.train()
            train_count = 0
            for inputs, labels in self.train_loader:

                # NOTE: 吃 class 的 index，所以用 argmax。
                labels = torch.argmax(labels.long(), dim=1)
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()
                output, h = self.model(inputs)
                loss = self.cri(output, labels)
                loss.backward()
                train_losses.append(loss.item())
                # accuracy = acc(output,labels)
                train_acc += (output.argmax(-1) == labels.float()).sum().item()
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip, error_if_nonfinite=True)
                self.opt.step()
                train_count += labels.size(0)

            val_losses = []
            val_acc = 0.0
            val_count = 0
            self.model.eval()
            with torch.no_grad():
                for inputs, labels in self.valid_loader:

                    labels = torch.argmax(labels.long(), dim=1)
                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)
                    output, val_h = self.model(inputs)
                    val_loss = self.cri(output, labels)
                    val_losses.append(val_loss.item())
                    # accuracy = acc(output,labels)
                    val_acc += (output.argmax(-1) ==
                                labels.float()).sum().item()
                    val_count += labels.size(0)

            epoch_train_loss = np.mean(train_losses)
            epoch_val_loss = np.mean(val_losses)
            epoch_train_acc = train_acc/train_count
            epoch_val_acc = val_acc/val_count
            epoch_tr_loss.append(epoch_train_loss)
            epoch_vl_loss.append(epoch_val_loss)
            epoch_tr_acc.append(epoch_train_acc)
            epoch_vl_acc.append(epoch_val_acc)
            print(f'Epoch {epoch+1}')
            print(
                f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
            print(
                f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')

            if epoch_val_acc >= self.valid_acc_min:
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
            print(25*'==')
        print('best valid acc', self.valid_acc_min)

    def eval(self):
        test_losses = []  # track loss
        num_correct = 0
        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        test_count = 0
        # iterate over test data
        for inputs, labels in self.test_loader:

            labels = torch.argmax(labels.long(), dim=1)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            output, test_h = self.model(inputs)
            # calculate loss
            test_loss = self.cri(output, labels)
            test_losses.append(test_loss.item())
            pred = output.argmax(-1)
            # correct_tensor = pred.eq(labels.float().view_as(pred))
            # correct = np.squeeze(correct_tensor.cpu().numpy())
            num_correct += (pred == labels.float()).sum().item()
            test_count += labels.size(0)

        print("Test loss: {:.3f}".format(np.mean(test_losses)))

        test_acc = num_correct/test_count
        print("Test accuracy: {:.3f}".format(test_acc))

    def tsne_(self):

        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        model = self.model.to(self.device)
        all_feats = []
        all_labels = []
        class_num = 0
        for inputs, labels in self.test_loader:

            with torch.no_grad():

                embeds = self.model.embedding(inputs)
                lstm_out, hidden = self.model.lstm(embeds)
                right = lstm_out[:, -1, :]

                for i in range(right.size(0)):
                    all_feats.append(right[i, :].numpy())
                    all_labels.append(labels[i, :].argmax(-1).item())

        tsne_plot_dir = self.save_dir[:-2]+'lstm.tsne.png'
        tsne(all_feats, all_labels, class_num, tsne_plot_dir)
        print('tsne saved in ', tsne_plot_dir)
