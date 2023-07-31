from os import read, urandom
import os
import math
import pandas as pd
from scipy.spatial.distance import euclidean
import numpy as np
from sklearn.manifold import TSNE
import random
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import matplotlib.patches as mpatches
import seaborn as sns
import os
sns.set()
sns.set_style('ticks')
from tensorflow.keras.models import load_model
import tensorflow as tf

from utils.util import *

params = {'legend.fontsize': 24,
        'figure.figsize': (20, 10),
        'axes.labelsize': 24,
        'axes.titlesize':24,
        'xtick.labelsize':24,
        'ytick.labelsize':18}
plt.rcParams.update(params)

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'

UTR_LEN = 128
DIM = 40
BATCH_SIZE = 100
gpath = '/home/sina/UTR/gan/logs/2023.07.21-22h39m31s_neo_july22_g5c5_d40_u128_utrdb2/checkpoint_h5/checkpoint_900.h5'
data_path = '/home/sina/UTR/data/utrdb2.csv'

def split_x_y(seqs):
    x = []
    y = []

    for index in range(len(seqs)):
        x.append(seqs[index][0])
        y.append(seqs[index][1])

    return x,y

def tsne(rand, real, gen, tsne_dim=2,use_rand=True):
    ax = axs[0]
    UTR_TSNE = TSNE(n_components=tsne_dim, perplexity=40, early_exaggeration=12, learning_rate=10,n_iter=5000,random_state=35, init='pca')
    size = int(len(rand[0])/4)

    if not use_rand:
        real = np.array(real)
        gen = np.array(gen)

        l_real = np.ones(len(real))
        l_gen = -np.ones(len(gen))

        all_ = np.concatenate((gen,real))
        labels = np.concatenate((l_gen,l_real))
    else:
        rand = np.array(rand)
        real = np.array(real)
        gen = np.array(gen)

        l_rand = np.zeros(len(rand))
        l_real = np.ones(len(real))
        l_gen = -np.ones(len(gen))

        all_ = np.concatenate((gen,rand,real))
        labels = np.concatenate((l_gen,l_rand,l_real))


    data = UTR_TSNE.fit_transform(all_)

    rand_x, rand_y = split_x_y(data)

    rgen = []
    rrand = []
    rreal = []

    for index in range(len(rand_y)):
        if labels[index] == 0:
            rrand.append((rand_x[index],rand_y[index]))
            ax.scatter(rand_x[index], rand_y[index],marker='.',s=50,  alpha=0.5, c='tab:green')
        elif labels[index] == 1:
            rreal.append((rand_x[index],rand_y[index]))
            ax.scatter(rand_x[index], rand_y[index], marker='.',s=50, alpha=0.5, c='tab:orange')
        elif labels[index] == -1:
            rgen.append((rand_x[index],rand_y[index]))
            ax.scatter(rand_x[index], rand_y[index],marker='.' ,s=50, alpha=0.5, c='tab:blue')
    

    ax.set_xlabel('tSNE-1')
    ax.set_ylabel('tSNE-2')

    return rgen, rrand, rreal


def umap_(rand, real, gen, umap_dim=2, use_rand=True,intron=False,display=False):
    ax = axs[1]
    UTR_UMAP = umap.UMAP(n_components=umap_dim,n_neighbors=20,min_dist=0.0)
    size = int(len(rand[0])/4)
    if not use_rand:
        real = np.array(real)
        gen = np.array(gen)

        l_real = np.ones(len(real))
        l_gen = -np.ones(len(gen))

        all_ = np.concatenate((gen,real))
        labels = np.concatenate((l_gen,l_real))
    else:
        rand = np.array(rand)
        real = np.array(real)
        gen = np.array(gen)

        l_rand = np.zeros(len(rand))
        l_real = np.ones(len(real))
        l_gen = -np.ones(len(gen))

        all_ = np.concatenate((gen,rand,real))
        labels = np.concatenate((l_gen,l_rand,l_real))

    data = UTR_UMAP.fit_transform(all_)

    rand_x, rand_y = split_x_y(data)
    
    rgen = []
    rrand = []
    rreal = []

    for index in range(len(rand_y)):
        if labels[index] == 0:
            rrand.append((rand_x[index],rand_y[index]))
            ax.scatter(rand_x[index], rand_y[index],marker='.',s=50,  alpha=0.5, c='tab:green')
        elif labels[index] == 1:
            rreal.append((rand_x[index],rand_y[index]))
            ax.scatter(rand_x[index], rand_y[index], marker='.',s=50, alpha=0.5, c='tab:orange')
        elif labels[index] == -1:
            rgen.append((rand_x[index],rand_y[index]))
            ax.scatter(rand_x[index], rand_y[index],marker='.' ,s=50, alpha=0.5, c='tab:blue')
    



    ax.set_xlabel('UMAP-1')
    ax.set_ylabel('UMAP-2')

    return rgen, rrand, rreal

def distances(rgen, rrand, rreal):
    gen_dists = []
    for i in range(len(rgen)):
        gen_dists.append(min(euclidean(c, rgen[i]) for c in rreal))

    

    rand_dists = []
    for i in range(len(rrand)):
        rand_dists.append(min(euclidean(c, rrand[i]) for c in rreal))

    return gen_dists, rand_dists

if __name__ == "__main__":
    random.seed(45)

    N = 200

    size = 78
    count = 0
    means_gen = []
    means_rand = []
    while size < 129:


        real = tsne_natural(data_path,size)

        enc_type = "one_hot"

        BATCH_SIZE = max(200, len(real))



        rand = random_data(length=size, size=BATCH_SIZE)
        gen = tsne_gen(path=gpath, length = size, BATCH_SIZE=BATCH_SIZE, DIM=DIM)

        enc_rand = encode(rand)
        enc_real = encode(real)
        enc_gen = encode(gen)

        # tsne
        tsne_dim = 2
        umap_dim = 2

        use_rand = True
        fig, axs = plt.subplots(1,2)

        rgen, rrand, rreal = tsne(enc_rand,enc_real,enc_gen,tsne_dim,use_rand)
        gen_dists, rand_dists = distances(rgen, rrand, rreal)
        # rgen, rrand, rreal = umap_(enc_rand,enc_real,enc_gen,umap_dim,use_rand)

        means_gen.append(np.average(gen_dists))
        means_rand.append(np.average(rand_dists))

        orange_patch = mpatches.Patch(color='tab:orange', label='Natural')
        blue_patch = mpatches.Patch(color='tab:blue', label='Generated')
        green_patch = mpatches.Patch(color='tab:green', label='Random')
        plt.legend(handles=[blue_patch,orange_patch,green_patch])

        axs[0].set_title('A',weight='bold',fontsize=24)
        gen_x = ['Generated' for i in range(len(gen_dists))]
        rand_x = ['Random' for i in range(len(rand_dists))]

        x = np.concatenate((gen_x,rand_x))
        y = np.concatenate((gen_dists,rand_dists))

        df = pd.DataFrame({'x':x,'y':y})

        sns.boxplot(x=df['x'],y=df['y'],ax=axs[1])
        axs[1].set_title('B',weight='bold',fontsize=24)

        fig.tight_layout()

        axs[1].set_ylabel("Min. Dist to Natural Data")
        axs[1].set_xlabel("")

        dir = '/home/sina/UTR/analysis/plots/tsne/'

        plt.savefig(dir+'tsne_umap_'+str(size)+'.png')

        size += 1

    with open('gen_mean.npy','wb') as f:
        np.save(f, np.array(means_gen))
    with open('rand_mean.npy','wb') as f:
        np.save(f, np.array(means_rand))