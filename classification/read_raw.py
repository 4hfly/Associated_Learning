import pandas as pd
from torch.utils.data import DataLoader, Dataset
import sentencepiece as spm
from torch.nn.utils.rnn import pad_sequence
import torch

from transformers import BertTokenizer

class IMDB(Dataset):
    
    def __init__(self, mode='train'):
        super().__init__()

        self.mode=mode
        
        # self.sp = spm.SentencePieceProcessor()
        # self.sp.load('m_bpe.model')
       
        self.tkr = BertTokenizer.from_pretrained("bert-base-uncased")
        print(self.tkr.encode("[PAD]"))
        self.read_data()

    def read_data(self):
        if self.mode == 'train':
            df = pd.read_excel('train.xlsx')
            reviews = df['Reviews'].tolist()
            input_data = []
            for s in reviews:
                try:
                    s = s.lower()
                    s = self.tkr.encode(s)['input_ids']
                    s = s[1:-1]
                    input_data.append(s)
                except:
                    input_data.append([0])
                    # reviews = [self.sp.encode(s) for s in reviews]
            self.input_data = input_data
            label = df['Sentiment'].tolist()
            mapp = {'neg':0, 'pos':1}
            self.label = [mapp[l] for l in label]
        else:
            df = pd.read_excel('test.xlsx')

            reviews = df['Reviews'].tolist()
            input_data = []
            for s in reviews:
                try:
                    s = s.lower()
                    s = self.tkr.encode(s)['input_ids'][1:-1]
                    input_data.append(s)
                except:
                    input_data.append([0])
                    # reviews = [self.sp.encode(s) for s in reviews]
            self.input_data = input_data
            label = df['Sentiment'].tolist()
            mapp = {'neg':0, 'pos':1}
            self.label = [mapp[l] for l in label]
        assert len(self.label) == len(self.input_data)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return torch.tensor(self.input_data[idx]), torch.tensor(self.label[idx])

    def collate(self, batch):
        text = [t for t,l in batch]
        label = [l for t,l in batch]
        text_tensor = pad_sequence(text, batch_first=True)
        label_tensor = torch.tensor(label)
        return text_tensor, label_tensor

