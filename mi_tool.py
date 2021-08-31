from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mutual_info_score

import numpy as np
# import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer

from scipy.stats import entropy
from scipy.stats import norm
from scipy.stats import binned_statistic

def tfidf(corpus):
    '''
    corpus: list of strings
    '''
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X


class MI_Vis:
    def __init__(self):
        self.name = 'Mutual information visulaization'
        self.l1_stack=[]
        self.l2_stack=[]
        self.l3_stack=[]
    def get_mi_yj(self, x, y):
        '''
        x: (n_samples, n_features)
        y; (n_samples, n_features)
        '''
        x_hid = x.shape[-1]
        y_hid = y.shape[-1]
        pt = PowerTransformer(method='box-cox')
        norm_x = pt.fit(x).transform(x)
        norm_y = pt.fit(y).transform(y)

        norm_x = norm.pdf(norm_x)
        norm_y = norm.pdf(norm_y)

        norm_x = np.mean(norm_x, axis=-1)
        norm_y = np.mean(norm_y, axis=-1)

        # print(norm_x[0], norm_y[0])
        mi = entropy(norm_x, norm_y, base=2)
        return mi
    
    def get_mi_bin(self, x, y, bins=30, eps=0.0001):

        '''
        x: (n_samples, n_features)
        y: (n_samples, n_features)
        
        todo: still have to design range(should be -1 to 1 if tanh)
        '''
        # bin for each x features
        bins = np.linspace(-1, 1, bins)
        num_x, feat_x = x.shape
        bin_x=[]
        for i in range(feat_x):
            d = np.digitize(x[:,i], bins=bins)
            uni, counts = np.unique(d, return_counts=True)
            map_ = dict(zip(uni, counts))
            d = [map_[j] for j in d]
            d = [eps+di for di in d]
            d = [di/sum(d) for di in d]
            bin_x.append(d)

        bin_x = np.array(bin_x)
        bin_x = np.transpose(bin_x)
        bin_x = np.mean(bin_x, axis=-1)

        num_y, feat_y = y.shape
        bin_y=[]
        for i in range(feat_y):
            d = np.digitize(y[:,i], bins=bins)
            uni, counts = np.unique(d, return_counts=True)
            map_ = dict(zip(uni, counts))
            d = [map_[j] for j in d]
            d = [eps+di for di in d]
            d = [di/sum(d) for di in d]
            bin_y.append(d)

        bin_y = np.array(bin_y)
        bin_y = np.transpose(bin_y)
        bin_y = np.mean(bin_y, axis=-1)
        
        mi = entropy(bin_x, bin_y, base=2)
        print(mi)
        


M = MI_Vis()
x = np.random.rand(4, 10)
t = np.random.rand(4, 20)

# x = np.ones((256,100))
# t = np.ones((256,200))

mi = M.get_mi_yj(x,t)
print(mi)
M.get_mi_bin(x,t, bins=30)
