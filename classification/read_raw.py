import gensim
import pandas as pd
import torch
import torch.nn as nn
import nltk
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.datasets import IMDB
from torchtext.vocab import GloVe


def build_vocab():

    data_iter = IMDB(split='train')
    lines = []
    for _, text in data_iter:
        lines.append(text)

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    lines = [line.lower().replace('\n', '').split() for line in lines]
    for i in range(len(lines)):
        for j in range(len(lines[i])):
            if lines[i][j] in stop_words:
                lines[i][j] = ''

    vocab = Dictionary(lines, prune_at=29998)
    special_tokens = {'<pad>': 0, '<unk>': 1}
    vocab.filter_extremes(keep_n=29998)
    vocab.patch_with_special_tokens(special_tokens)
    # vocab.filter_extremes(keep_n=24998)

    print(len(vocab))               # 30000
    print(vocab.token2id['<pad>'])  # 0
    print(vocab.token2id['great'])  # 798
    vocab.save('imdb_vocab.pkl')


def get_emb(vocab):
    vectors = GloVe(name='6B', dim=300)
    w = []
    for word in vocab.token2id.keys():
        try:
            w.append(vectors[word])
        except:
            w.append(torch.zeros(300))
    w = torch.stack(w, dim=0)
    return nn.Embedding.from_pretrained(w, freeze=False, padding_idx=0)


# build_vocab()


class IMDB(Dataset):

    def __init__(self, mode='train'):
        super().__init__()

        self.mode = mode
        self.vocab = Dictionary.load('imdb_vocab.pkl')
        self.stop_words = set(stopwords.words('english'))
        self.read_data()

    def read_data(self):

        if self.mode == 'train':
            df = pd.read_excel('train.xlsx')
            reviews = df['Reviews'].tolist()
            input_data = []
            for s in reviews:
                ids = []
                if type(s) == str:
                    s = s.lower()
                else:
                    s = '<pad> <pad>'
                for tok in s.split():
                    if tok in self.stop_words:
                        continue
                    try:
                        ids.append(self.vocab.token2id[tok])
                    except:
                        ids.append(self.vocab.token2id['<unk>'])
                input_data.append(ids)

            self.input_data = input_data
            label = df['Sentiment'].tolist()
            mapp = {'neg': 0, 'pos': 1}
            self.label = [mapp[l] for l in label]
        else:
            df = pd.read_excel('test.xlsx')

            reviews = df['Reviews'].tolist()
            input_data = []

            for s in reviews:
                ids = []
                if type(s) == str:
                    s = s.lower()
                else:
                    s = '<pad> <pad>'
                for tok in s.split():
                    if tok in self.stop_words:
                        continue
                    try:
                        ids.append(self.vocab.token2id[tok])
                    except:
                        ids.append(self.vocab.token2id['<unk>'])
                input_data.append(ids)

            self.input_data = input_data
            label = df['Sentiment'].tolist()
            mapp = {'neg': 0, 'pos': 1}
            self.label = [mapp[l] for l in label]
        assert len(self.label) == len(self.input_data)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.tensor(self.input_data[idx]), torch.tensor(self.label[idx])

    def collate(self, batch):
        text = [t for t, l in batch]
        label = [l for t, l in batch]
        text_tensor = pad_sequence(text, batch_first=True)
        label_tensor = torch.tensor(label)
        return text_tensor, label_tensor
