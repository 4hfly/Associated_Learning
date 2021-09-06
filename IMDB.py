import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from classification.model import EmbeddingAL, LSTMAL
from utils import *
import pandas as pd
import os
os.environ["WANDB_SILENT"] = "true"

parser = argparse.ArgumentParser('IMDB Dataset for AL training')

# model param
parser.add_argument('--word-emb', type=int,
                    help='word embedding dimension', default=300)
parser.add_argument('--label-emb', type=int,
                    help='label embedding dimension', default=128)
parser.add_argument('--l1-dim', type=int,
                    help='lstm1 hidden dimension', default=400)
parser.add_argument('--bridge-dim', type=int,
                    help='bridge function dimension', default=300)
parser.add_argument('--vocab-size', type=int, help='vocab-size', default=30000)

parser.add_argument('--pretrain-emb', type=str, help='pretrained word embedding: glove or fasttest', default='glove')
parser.add_argument('--data-position', type=str, help='shuffle or sort', default='shuffle')
# training param
parser.add_argument('--lr', type=float, help='lr', default=0.001)
parser.add_argument('--batch-size', type=int, help='batch-size', default=64)
parser.add_argument('--one-hot-label', type=bool,
                    help='if true then use one-hot vector as label input, else integer', default=True)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--class-num', type=int, default=2)
# dir param
parser.add_argument('--save-dir', type=str, default='ckpt/imdb.pt')

parser.add_argument('--act', type=str,
                    default='tanh')

args = parser.parse_args()
df = pd.read_csv('IMDB_Dataset.csv')
df['cleaned_reviews'] = df['review'].apply(data_preprocessing, True)

corpus = [word for text in df['cleaned_reviews'] for word in text.split()]
train_text = df['cleaned_reviews'].tolist()
vocab = create_vocab(corpus, args.vocab_size)

print('vocab size',len(vocab))

clean_train = df['cleaned_reviews'].tolist()
print(len(clean_train))
clean_train_id = convert2id(clean_train, vocab)

df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

train_label = df['sentiment'].tolist()
train_label = multi_class_process(train_label, 2)

train_features = Padding(clean_train_id, 400)
X_train, X_remain, y_train, y_remain = train_test_split(train_features, train_label, test_size=0.2, random_state=1)
X_valid, X_test, y_valid, y_test = train_test_split(X_remain, y_remain, test_size=0.5, random_state=1)

print('dataset information:')
print('=====================')
print('train size', len(X_train))
print('valid size', len(X_valid))
print('test size', len(X_test))
print('=====================\n')


# create tensor dataset
train_data = TensorDataset(torch.from_numpy(X_train), torch.stack(y_train))
test_data = TensorDataset(torch.from_numpy(X_test), torch.stack(y_test))
valid_data = TensorDataset(torch.from_numpy(X_valid), torch.stack(y_valid))

batch_size = args.batch_size

if args.data_position == 'sort':
    train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
else:
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=False, batch_size=batch_size)


class Cls(nn.Module):
    
    def __init__(
        self, vocab_size, embedding_dim, hidden_dim, n_layers, class_num, drop_prob=0.1, pretrain=None
    ):

        super(Cls, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        if pretrain is not None:
            self.embedding = nn.Embedding.from_pretrained(pretrain, freeze=False, padding_idx=0)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 400),
            nn.Tanh(),
            nn.Linear(400, class_num)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):

        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds)
        lstm_out = lstm_out[:, -1, :]
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.softmax(out)
        return out, hidden

act = get_act(args)

torch.cuda.empty_cache()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.pretrain_emb == 'none': 
    model = Cls(args.vocab_size, args.word_emb, args.l1_dim, 2, args.class_num)
else:
    w = get_word_vector(vocab, emb=args.pretrain_emb)
    model = Cls(args.vocab_size, args.word_emb, args.l1_dim, 2, args.class_num, pretrain=w)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
model = model.to(device)
print('IMDB LSTM model param num', get_n_params(model))
T = Trainer(model, args.lr, train_loader=train_loader,
              valid_loader=valid_loader, test_loader=test_loader, save_dir=args.save_dir)
T.run(epochs=args.epoch)
T.eval()
T.tsne_()