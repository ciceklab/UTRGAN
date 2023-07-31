from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from scipy.stats import ks_2samp,kstest,ttest_ind, mannwhitneyu, norm
from cliffs_delta import cliffs_delta
import seaborn as sns
from tqdm import tqdm
import random
random.seed(1337)
import os
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None
import RNA
from polyleven import levenshtein
import time
import itertools
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

from utils.util import *
from utils.framepool import *

colors = ["#3c5068", "#acbab6", "#dcd3cd", "#d4a6a6"]

tf.compat.v1.enable_eager_execution()

# os.environ["CUDA_VISIBLE_DEVICES"] = '5'

def ES_CI(d1, d2):
    n1 = len(d1)
    n2 = len(d2)

    u1 = np.mean(d1)
    u2 = np.mean(d2)
    
    s1 = np.std(d1)
    s2 = np.std(d2)

    s = np.sqrt(((n1 - 1) * np.power(s1,2) + (n2 - 1) * np.power(s2,2)) / (n1 + n2 - 2))

    effect_size = (u1 - u2)/s
    effect_size = cliffs_delta(d1, d2)



    ct1 = n1  #items in dataset 1
    ct2 = n2  #items in dataset 2
    ds1 = d1
    ds2 = d2
    alpha = 0.05       #95% confidence interval
    N = norm.ppf(1 - alpha/2) # percent point function - inverse of cdf

    # The confidence interval for the difference between the two population
    # medians is derived through these nxm differences.
    diffs = sorted([i-j for i in ds1 for j in ds2])

    # For an approximate 100(1-a)% confidence interval first calculate K:
    k = int(round(ct1*ct2/2 - (N * (ct1*ct2*(ct1+ct2+1)/12)**0.5)))

    # The Kth smallest to the Kth largest of the n x m differences 
    # ct1 and ct2 should be > ~20
    CI = (diffs[k], diffs[len(diffs)-k])


    return effect_size, CI


customPalette = {'Generated':colors[0],'Random':colors[3],'Natural':colors[1]}

UTR_LEN = 128
Z_DIM = 40
DIM = Z_DIM
BATCH_SIZE = 2048
MAX_LEN = UTR_LEN
gpath = '/home/sina/UTR/gan/logs/2023.07.21-22h39m31s_neo_july22_g5c5_d40_u128_utrdb2/checkpoint_h5/checkpoint_3000.h5'
data_path = '/home/sina/UTR/data/utrdb2.csv'
mrl_path = '/home/sina/UTR/models/utr_model_combined_residual_new.h5'



sns.set()
sns.set_style('ticks')

params = {'legend.fontsize': 32,
        'figure.figsize': (54, 40),
        'axes.labelsize': 54,
        'axes.titlesize':54,
        'xtick.labelsize':54,
        'ytick.labelsize':36}

#POSTER
params = {'legend.fontsize': 48,
        'figure.figsize': (54, 32),
        'axes.labelsize': 60,
        'axes.titlesize':60,
        'xtick.labelsize':60,
        'ytick.labelsize':60}

plt.rcParams.update(params)


model = load_framepool(mrl_path)

gens = generate_data(path=gpath, UTR_LEN=UTR_LEN, BATCH_SIZE=BATCH_SIZE, DIM=DIM)
gens2 = generate_data(path=gpath, UTR_LEN=UTR_LEN, BATCH_SIZE=4096, DIM=DIM)
gens_encoded = np.array([encode_seq_framepool(seq) for seq in gens])
randoms = random_data(length=UTR_LEN, size=BATCH_SIZE)
randoms_encoded = np.array([encode_seq_framepool(seq) for seq in randoms])
naturals = read_real(data_path, UTR_LEN=UTR_LEN, all=False, samples=BATCH_SIZE)
naturals2 = read_real(data_path, UTR_LEN=UTR_LEN, all=False, samples=10000)
naturals_encoded = np.array([encode_seq_framepool(seq) for seq in naturals])
naturals_all = read_real(data_path, UTR_LEN=UTR_LEN, all=True)
naturals_encoded_all = np.array([encode_seq_framepool(seq) for seq in naturals_all])
############################# MRL PREDICTION ####################################

######### Gens

gens_tensor = tf.convert_to_tensor(gens_encoded,dtype=tf.float32)
pred_gens = model(gens_tensor)
pred_gens = tf.reshape(pred_gens,(-1))
genpreds = pred_gens.numpy().astype('float')

######### Randoms

randoms_tensor = tf.convert_to_tensor(randoms_encoded,dtype=tf.float32)
pred_randoms = model(randoms_tensor)
pred_randoms = tf.reshape(pred_randoms,(-1))
randpreds = pred_randoms.numpy().astype('float')

######## Labeled

naturals_tensors = tf.convert_to_tensor(naturals_encoded_all,dtype=tf.float32)
pred_naturals = model(naturals_tensors)
pred_naturals = tf.reshape(pred_naturals,(-1))
realpreds = pred_naturals.numpy().astype('float')

############

bins = np.linspace(2.5, 9, 30)

fig, axs = plt.subplots(2,3)    

real_x = ['Natural' for i in range(len(realpreds))]
gen_x = ['Generated' for i in range(len(genpreds))]
rand_x = ['Random' for i in range(len(randpreds))]

x = np.concatenate((gen_x,real_x,rand_x))
y = np.concatenate((genpreds,realpreds,randpreds))

gent_mrl = ttest_ind(genpreds,realpreds)
randt_mrl = ttest_ind(randpreds,realpreds)
genu_mrl = mannwhitneyu(genpreds, realpreds)
randu_mrl = mannwhitneyu(randpreds, realpreds)
es_gen_mrl = ES_CI(genpreds,realpreds)
es_rand_mrl = ES_CI(randpreds,realpreds)

df = pd.DataFrame({'x':x,'y':y})

sns.violinplot(x=df['x'],y=df['y'],ax=axs[1,0],palette=customPalette)

axs[1,0].set_ylabel("Mean Ribosome Load")
axs[1,0].set_xlabel("")

############################# MFE PREDICTION ####################################

genpreds = []

for i in range(len(gens)):
    (ss, mfe) = RNA.fold(gens[i])
    genpreds.append(mfe)

randpreds = []

for i in range(len(randoms)):
    (ss, mfe) = RNA.fold(randoms[i])
    randpreds.append(mfe)

realpreds = []

for i in range(len(naturals_all)):
    (ss, mfe) = RNA.fold(naturals_all[i])
    realpreds.append(mfe)

real_x = ['Natural' for i in range(len(realpreds))]
gen_x = ['Generated' for i in range(len(genpreds))]
rand_x = ['Random' for i in range(len(randpreds))]

gent_mfe = ttest_ind(genpreds,realpreds)
randt_mfe = ttest_ind(randpreds,realpreds)
randu_mfe = mannwhitneyu(randpreds, realpreds)
genu_mfe = mannwhitneyu(genpreds, realpreds)
es_gen_mfe = ES_CI(genpreds,realpreds)
es_rand_mfe = ES_CI(randpreds,realpreds)

x = np.concatenate((gen_x,real_x,rand_x))
y = np.concatenate((genpreds,realpreds,randpreds))

df = pd.DataFrame({'x':x,'y':y})

sns.violinplot(x=df['x'],y=df['y'],ax=axs[1,2],palette=customPalette)

axs[1,2].set_ylabel("Minimum Free Energy")
axs[1,2].set_xlabel("")

############################# Levenshtein Distance ####################################



DIST = 'S'

if DIST == 'KMER':
    dist_rand = calc_dist_kmer(randoms, naturals)
    dist_gen = calc_dist_kmer(gens, naturals)
    dist_real = calc_dist_kmer(naturals, naturals)
else:
    if os.path.exists('/home/sina/UTR/analysis/plots/rand_ham_new.npy'):
        dist_rand = np.load('/home/sina/UTR/analysis/plots/rand_ham_new.npy', allow_pickle=True)
    else:
        dist_rand = hamming_dist(randoms,naturals_all)
        with open("/home/sina/UTR/analysis/plots/rand_ham_new.npy", 'wb') as f:
            np.save(f,dist_rand)

    if os.path.exists('/home/sina/UTR/analysis/plots/real_ham_new.npy'):
        dist_real = np.load('/home/sina/UTR/analysis/plots/real_ham_new.npy', allow_pickle=True)
    else:
        dist_real = hamming_dist(naturals, naturals_all)
        with open("/home/sina/UTR/analysis/plots/real_ham_new.npy", 'wb') as f:
            np.save(f,dist_real)        

    if os.path.exists('/home/sina/UTR/analysis/plots/gen_ham_new.npy'):
        dist_gen = np.load('/home/sina/UTR/analysis/plots/gen_ham_new.npy', allow_pickle=True)
    else:
        dist_gen = hamming_dist(gens, naturals_all)
        with open("/home/sina/UTR/analysis/plots/gen_ham_new.npy", 'wb') as f:
            np.save(f,dist_gen)

# filter:
dist_real_filtered = []
for i in range(len(dist_real)):
    if dist_real[i] > 21:
        dist_real_filtered.append(dist_real[i])


dist_real = dist_real_filtered

real_x = ['Natural' for i in range(len(dist_real))]
gen_x = ['Generated' for i in range(len(dist_gen))]
rand_x = ['Random' for i in range(len(dist_rand))]

gent_dist = ttest_ind(dist_gen,dist_real)
randt_dist = ttest_ind(dist_rand,dist_real)
genu_dist = mannwhitneyu(dist_gen, dist_real)
randu_dist = mannwhitneyu(dist_rand, dist_real)
es_gen_lev = ES_CI(dist_gen,dist_real)
es_rand_lev = ES_CI(dist_rand,dist_real)

x = np.concatenate((gen_x,real_x,rand_x))
y = np.concatenate((dist_gen,dist_real,dist_rand))

df = pd.DataFrame({'x':x,'y':y})

sns.violinplot(x=df['x'],y=df['y'],ax=axs[0,0], palette=customPalette)

if DIST == 'KMER':

    
    axs[0,0].set_ylabel("Min. 4-mer Distance")
    axs[0,0].set_xlabel("")

else:

    axs[0,0].set_ylabel("Min. Levenshtein Distance")
    axs[0,0].set_xlabel("")

############################################################################

if os.path.exists('/home/sina/UTR/analysis/plots/rand_4mer_new.npy'):
    dist_rand = np.load('/home/sina/UTR/analysis/plots/rand_4mer_new.npy', allow_pickle=True)
else:
    dist_rand = calc_dist_kmer(randoms, naturals_all)
    with open("/home/sina/UTR/analysis/plots/rand_4mer_new.npy", 'wb') as f:
        np.save(f,dist_rand)

if os.path.exists('/home/sina/UTR/analysis/plots/real_4mer_new.npy'):
    dist_real = np.load('/home/sina/UTR/analysis/plots/real_4mer_new.npy', allow_pickle=True)
else:
    dist_real = calc_dist_kmer(naturals, naturals_all)
    with open("/home/sina/UTR/analysis/plots/real_4mer_new.npy", 'wb') as f:
        np.save(f,dist_real)        

if os.path.exists('/home/sina/UTR/analysis/plots/gen_4mer_new.npy'):
    dist_gen = np.load('/home/sina/UTR/analysis/plots/gen_4mer_new.npy', allow_pickle=True)
else:
    dist_gen = calc_dist_kmer(gens, naturals_all)
    with open("/home/sina/UTR/analysis/plots/gen_4mer_new.npy", 'wb') as f:
        np.save(f,dist_gen)

anomalies = 0

dist_real_filtered = []
for i in range(len(dist_real)):
    if dist_real[i] > 7.5:
        dist_real_filtered.append(dist_real[i])
    else:
        anomalies += 1

print(anomalies)

dist_real = dist_real_filtered

real_x = ['Natural' for i in range(len(dist_real))]
gen_x = ['Generated' for i in range(len(dist_gen))]
rand_x = ['Random' for i in range(len(dist_rand))]

gent_dist2 = ttest_ind(dist_gen,dist_real)
randt_dist2 = ttest_ind(dist_rand,dist_real)
genu_dist2 = mannwhitneyu(dist_gen, dist_real)
randu_dist2 = mannwhitneyu(dist_rand, dist_real)
es_gen_4mer = ES_CI(dist_gen,dist_real)
es_rand_4mer = ES_CI(dist_rand,dist_real)

x = np.concatenate((gen_x,real_x,rand_x))
y = np.concatenate((dist_gen,dist_real,dist_rand))

df = pd.DataFrame({'x':x,'y':y})

sns.violinplot(x=df['x'],y=df['y'],ax=axs[0,1], palette = customPalette)


axs[0,1].set_ylabel("Min. 4-mer Distance")
axs[0,1].set_xlabel("")


############################# GC Content ####################################

rand_gc = get_gc_content_many(randoms)
real_gc = get_gc_content_many(naturals_all)
gens_gc = get_gc_content_many(gens)

real_x = ['Natural' for i in range(len(real_gc))]
gen_x = ['Generated' for i in range(len(gens_gc))]
rand_x = ['Random' for i in range(len(rand_gc))]

x = np.concatenate((gen_x,real_x,rand_x))
y = np.concatenate((gens_gc,real_gc,rand_gc))

gent_gc = ttest_ind(gens_gc, real_gc)
randt_gc = ttest_ind(rand_gc, real_gc)
genu_gc = mannwhitneyu(gens_gc, real_gc)
randu_gc = mannwhitneyu(rand_gc, real_gc)
es_gen_gc = ES_CI(gens_gc,real_gc)
es_rand_gc = ES_CI(rand_gc,real_gc)

df = pd.DataFrame({'x':x,'y':y})

sns.violinplot(x=df['x'],y=df['y'],ax=axs[0,2], palette=customPalette)

axs[0,2].set_ylabel("G/C Content")
axs[0,2].set_xlabel("")

########################################################### TE

randpreds = np.load('/home/sina/UTR/optimization/mrl/te_rands.npy',allow_pickle=True)
genpreds = np.load('/home/sina/UTR/optimization/mrl/te_gens.npy',allow_pickle=True)
realpreds = np.load('/home/sina/UTR/optimization/mrl/te_reals.npy',allow_pickle=True)

real_x = ['Natural' for i in range(len(realpreds))]
gen_x = ['Generated' for i in range(len(genpreds))]
rand_x = ['Random' for i in range(len(randpreds))]

x = np.concatenate((gen_x,real_x,rand_x))
y = np.concatenate((genpreds,realpreds,randpreds))

gent_te = ttest_ind(genpreds, realpreds)
randt_te = ttest_ind(randpreds, realpreds)
genu_te = mannwhitneyu(genpreds, realpreds)
randu_te = mannwhitneyu(randpreds, realpreds)
es_gen_te = ES_CI(genpreds,realpreds)
es_rand_te = ES_CI(randpreds,realpreds)

df = pd.DataFrame({'x':x,'y':y})

sns.violinplot(x=df['x'],y=df['y'],ax=axs[1,1], palette=customPalette)

axs[1,1].set_ylabel("Translation Efficiency")
axs[1,1].set_xlabel("")

############################################################

axs[1,0].set_title('D',weight='bold',fontsize=64,loc='left')
axs[1,1].set_title('E',weight='bold',fontsize=64,loc='left')
axs[1,2].set_title('F',weight='bold',fontsize=64,loc='left')
axs[0,0].set_title('A',weight='bold',fontsize=64,loc='left')
axs[0,1].set_title('B',weight='bold',fontsize=64,loc='left')
axs[0,2].set_title('C',weight='bold',fontsize=64,loc='left')
fig.tight_layout(pad=2)

plt.savefig('./plots/violins_all.png')

print("Mean Ribosome Load KStest:")
print(gent_mrl)
print(genu_mrl)
print(es_gen_mrl)
print(randt_mrl)
print(randu_mrl)
print(es_rand_mrl)
print("Minimum Free Energy KStest:")
print(gent_mfe)
print(genu_mfe)
print(es_gen_mfe)
print(randt_mfe)
print(randu_mfe)
print(es_rand_mfe)
print("Ham Distance KStest:")
print(gent_dist)
print(genu_dist)
print(es_gen_lev)
print(randt_dist)
print(randu_dist)
print(es_rand_lev)
print("4mer Distance KStest:")
print(gent_dist2)
print(genu_dist2)
print(es_gen_4mer)
print(randt_dist2)
print(randu_dist2)
print(es_rand_4mer)
print("GC Content KStest:")
print(gent_gc)
print(genu_gc)
print(es_gen_gc)
print(randt_gc)
print(randu_gc)
print(es_rand_gc)
print("TE KStest:")
print(gent_te)
print(genu_te)
print(es_gen_te)
print(randt_te)
print(randu_te)
print(es_rand_te)