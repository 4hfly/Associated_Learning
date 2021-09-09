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

from prettytable import PrettyTable
from mi_tool import MI_Vis

import wandb


class L1_ALTrainer:

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
        wandb.init(project=project_name, entity='al-train')
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

        self.ckpt_epoch = 0
        self.pat = 0

    def run(self, epoch):

        # TODO: ALTrainer 和 Trainer 這兩個的 epoch 一個有 s，一個沒有，應該要統一一下。
        self.valid_acc_min = -999
        for epoch in range(epoch):
            train_losses = []

            total_emb_loss = []
            total_l1_loss = []

            total_acc = 0.0
            total_count = 0
            self.model.train()

            # initialize hidden state
            for inputs, labels in self.train_loader:

                self.model.train()
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.opt.zero_grad()

                emb_loss, l1_loss = self.model(inputs, labels)

                wandb.log({"emb loss": emb_loss.item()})
                wandb.log({"lstm1 loss": l1_loss.item()})
                wandb.log({"emb bridge loss": self.model.embedding.loss_b})
                wandb.log({"emb decode loss": self.model.embedding.loss_d})
                wandb.log({"lstm1 bridge loss": self.model.layer_1.loss_b})
                wandb.log({"lstm1 decode loss": self.model.layer_1.loss_d})

                loss = emb_loss + l1_loss
                wandb.log({"total loss": loss.item()})
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.model.layer_1.parameters(), 5, error_if_nonfinite=True)

                total_emb_loss.append(emb_loss.item())
                total_l1_loss.append(l1_loss.item())

                self.opt.step()

                torch.cuda.empty_cache()

                # calculate the loss and perform backprop
                with torch.no_grad():

                    self.model.eval()

                    left = self.model.embedding.f(inputs)
                    # left = self.model.ln(left)

                    output, hidden = self.model.layer_1.f(left)
                    # output = self.model.ln2(output)
                    left = output[:, -1, :]
                    right = self.model.layer_1.bx(left)
                    right = self.model.layer_1.dy(right)

                    predicted_label = self.model.embedding.dy(right)

                    total_acc += (predicted_label.argmax(-1) ==
                                    labels.to(torch.float).argmax(-1)).sum().item()

                    total_count += labels.size(0)

            # TODO: 這個 val_losses 沒用到。
            val_losses = []
            val_acc = 0.0
            val_count = 0

            self.model.eval()

            for inputs, labels in self.valid_loader:

                with torch.no_grad():

                    self.model.embedding.eval()
                    self.model.layer_1.eval()

                    inputs, labels = inputs.to(
                        self.device), labels.to(self.device)

                    left = self.model.embedding.f(inputs)
                    output, hidden = self.model.layer_1.f(left)
                    left = output[:,-1,:]


                    right = self.model.layer_1.bx(left)
                    right = self.model.layer_1.dy(right)

                    predicted_label = self.model.embedding.dy(right)

                    val_acc += (predicted_label.argmax(-1) ==
                                labels.to(torch.float).argmax(-1)).sum().item()

                    val_count += labels.size(0)

            epoch_train_loss = [np.mean(total_emb_loss), np.mean(
                total_l1_loss)]
            epoch_train_acc = total_acc/total_count
            epoch_val_acc = val_acc/val_count

            print(f'Epoch {epoch+1}')
            print(
                f'train_loss : emb loss {epoch_train_loss[0]}, layer1 loss {epoch_train_loss[1]}')
            print(
                f'train_accuracy : {epoch_train_acc*100} val_accuracy : {epoch_val_acc*100}')
            wandb.log({"train acc": epoch_train_acc})
            wandb.log({"valid acc": epoch_val_acc})

            if epoch_val_acc >= self.valid_acc_min:
                self.pat = 0
                self.ckpt_epoch = epoch+1
                torch.save(self.model.state_dict(), f'{self.save_dir}')
                print('Validation acc increased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    self.valid_acc_min, epoch_val_acc))
                self.valid_acc_min = epoch_val_acc
            print(25*'==')
        final_dp = self.save_dir[:-3] + 'last.pth'
        torch.save(self.model.state_dict(), f'{final_dp}')
        print('best val acc', self.valid_acc_min)
        print('best checkpoint at', self.ckpt_epoch, 'epoch')

    def write_pred(self):
        test_losses = []  # track loss
        num_correct = 0
        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        test_count = 0
        pred_list=[]
        # iterate over test data
        for inputs, labels in self.test_loader:

            left = model.embedding.f(inputs)
            output, hidden = model.layer_1.f(left)
            left = output[:, -1, :]
            right = model.layer_1.bx(left)
            right = model.layer_1.dy(right)

            pred = self.model.embedding.dy(right).argmax(-1)
            pred_list = pred_list + pred.tolist()
        tsv_name = self.save_dir[:-3]+'.tsv'
        df = {'index':[i for i in range(len(pred_list))], 'prediction':pred_list}
        df = pd.DataFrame(df)
        df.to_csv(tsv_name, sep='\t',index=False)

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

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                left = model.embedding.f(inputs)
                output, hidden = model.layer_1.f(left)
                left = output[:, -1, :]
                right = model.layer_1.bx(left)
                right = model.layer_1.dy(right)

                predicted_label = self.model.embedding.dy(right)

                test_acc += (predicted_label.argmax(-1) ==
                                labels.to(torch.float).argmax(-1)).sum().item()

                test_count += labels.size(0)
        x = test_acc/test_count
        wandb.log({"test acc": x})
        print('Test acc', test_acc/test_count)
        # self.tsne_()

    def tsne_(self):
        print('working on tsne')
        self.model.eval()
        self.model.load_state_dict(torch.load(f'{self.save_dir}'))
        self.model = self.model.to(self.device)
        all_feats = []
        all_labels = []
        class_num = 0
        for inputs, labels in self.test_loader:

            with torch.no_grad():

                self.model.embedding.eval()
                self.model.layer_1.eval()
                self.model.layer_2.eval()

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                inputs, labels = inputs.to(self.device), labels.to(self.device)

                left = self.model.embedding.f(inputs)
                output, hidden = self.model.layer_1.f(left)
                left, (output, c) = self.model.layer_2.f(output, hidden)
                left = left[:, -1, :]
                # left = left.reshape(left.size(1), -1)
                right = self.model.layer_2.bx(left)
                right = self.model.layer_2.dy(right)
                right = right.cpu()
                labels = labels.cpu()
                # left = left.reshape(left.size(1), -1)
                for i in range(right.size(0)):
                    all_feats.append(right[i, :].numpy())
                    all_labels.append(labels[i, :].argmax(-1).item())

        tsne_plot_dir = self.save_dir[:-2]+'al2.tsne.png'
        tsne(all_feats, all_labels, class_num, tsne_plot_dir)
        # print('tsne saved in ', tsne_plot_dir)