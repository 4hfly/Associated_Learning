from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

def tsne(x,y, class_num, save_dir):
    '''
    x: (n_sample, n_feature)
    y: (n_sample, )
    class_num: int
    '''
    x_emb = TSNE(n_components=2, random_state=0, perplexity=15).fit_transform(x)
    x_emb_x = x_emb[:,0]
    x_emb_y = x_emb[:,1]
    plt.scatter(x_emb_x, x_emb_y, c=y)
    plt.savefig(save_dir)

