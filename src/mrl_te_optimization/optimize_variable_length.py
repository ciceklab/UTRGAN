
from tqdm import tqdm
from lib import utils
import socket
import datetime
from importlib import reload
import re
import itertools
from pathlib import Path
import random
random.seed(1337)
import os
import pickle
from decimal import Decimal
import collections
import matplotlib.pyplot as plt
import argparse
# numpy and similar
import numpy as np
np.random.seed(1337)
import pandas as pd
pd.options.mode.chained_assignment = None 
import scipy.stats as stats

# Deep Learning packages
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D, Dropout, Concatenate, Lambda, Flatten, ZeroPadding1D, MaxPooling1D, BatchNormalization, ThresholdedReLU, Masking, Add, LSTM, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import losses
from tensorflow.keras.utils import Sequence

tf.compat.v1.enable_eager_execution()

from Bio import SeqIO
import pandas as pd
import numpy as np
import requests, sys


parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, required=False ,default='./../../data/utrs.csv')    
parser.add_argument('-uc', type=int, required=False ,default=64)
parser.add_argument('-lr', type=int, required=False ,default=5)
parser.add_argument('-gpu', type=str, required=False ,default='-1')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def fetch_seq(start, end, chr, strand):
    server = "https://rest.ensembl.org"
    
    ext = "/sequence/region/human/" + str(chr) + ":" + str(start) + ".." + str(end) + ":" + str(strand) +  "?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    return r.text

def apply_pad_mask(input_tensors):
    tensor = input_tensors[0]
    mask = input_tensors[1]
    mask = K.expand_dims(mask, axis=2)
    return tf.multiply(tensor, mask)

# Layer which slices input tensor into three tensors, one for each frame w.r.t. the canonical start
class FrameSliceLayer(Layer):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape) 
    
    def call(self, x):
        shape = K.shape(x)
        x = K.reverse(x, axes=1) # reverse, so that frameness is related to fixed point (start codon)
        frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
        frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
        frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)
        return [frame_1, frame_2, frame_3]
    
    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            return [(input_shape[0], None),(input_shape[0], None),(input_shape[0], None)]
        return [(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2])]

def convert_model(model_:Model):
    print(model_.summary())
    input_ = tf.keras.layers.Input(shape=( 10500, 4))
    input = input_
    for i in range(len(model_.layers)-1):
        print(type(model_.layers[i+1]))

        if isinstance(model_.layers[i+1],FrameSliceLayer):
            a = FrameSliceLayer()
            output = a(input)
            input = output

        else:
        
            if isinstance(model_.layers[i+1],tf.keras.layers.Concatenate):
                paddings = tf.constant([[0,0],[0,6]])
                output = tf.pad(input, paddings, 'CONSTANT')
                input = output
            else:
                if not isinstance(model_.layers[i+1],tf.keras.layers.InputLayer):
                    output = model_.layers[i+1](input)
                    input = output

                if isinstance(model_.layers[i+1],tf.keras.layers.Conv1D):
                    pass

            i

    model = tf.keras.Model(inputs=input_, outputs=output)
    model.compile(loss="mse", optimizer="adam")
    print(model.summary())
    return model

def one_hot(seq):
    convert = False
    if isinstance(seq, tf.Tensor):
        seq = seq.numpy().astype(str)
        convert = True

    num_seqs = len(seq)
    seq_len = len(seq[0])
    seqindex = {'A':0, 'C':1, 'G':2, 'T':3, 'a':0, 'c':1, 'g':2, 't':3}
    seq_vec = np.zeros((num_seqs,seq_len,4), dtype='bool')
    for i in range(num_seqs):
        thisseq = seq[i]
        for j in range(seq_len):
            try:
                seq_vec[i,j,seqindex[thisseq[j]]] = 1
            except:
                pass
    
    if convert:
        seq_vec = tf.convert_to_tensor(seq_vec,dtype=tf.float32)


    return seq_vec

def convert_gradient(mrl_gradient):
    pass

def recover_seq(samples, rev_charmap):
    """Convert samples to strings and save to log directory."""
    if isinstance(samples,tf.Tensor):
        samples = samples.numpy()

    char_probs = samples
    argmax = np.argmax(char_probs, 2)
    seqs = []
    for line in argmax:
        s = "".join(rev_charmap[d] for d in line)
        s = s.replace('*','')
        seqs.append(s)

    seqs = np.array(seqs)
    return seqs   

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def log(samples_dir=False):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs/", "gan_test_opt", stamp)

    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    return full_logdir, 0

if __name__ == '__main__':
    


    logdir, checkpoint_baseline = log(samples_dir=True)

    # Layer which slices input tensor into three tensors, one for each frame w.r.t. the canonical start
    class FrameSliceLayer(Layer):
        
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def build(self, input_shape):
            super().build(input_shape) 
        
        def call(self, x):
            shape = K.shape(x)
            x = K.reverse(x, axes=1) # reverse, so that frameness is related to fixed point (start codon)
            frame_1 = tf.gather(x, K.arange(start=0, stop=shape[1], step=3), axis=1)
            frame_2 = tf.gather(x, K.arange(start=1, stop=shape[1], step=3), axis=1)
            frame_3 = tf.gather(x, K.arange(start=2, stop=shape[1], step=3), axis=1)
            return [frame_1, frame_2, frame_3]
        
        def compute_output_shape(self, input_shape):
            if len(input_shape) == 2:
                return [(input_shape[0], None),(input_shape[0], None),(input_shape[0], None)]
            return [(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2]),(input_shape[0], None, input_shape[2])]


    min_len = None
    nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 
                    'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 
                    'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}

    MAX_LEN = 128
        
    def encode_seq(seq, max_len=MAX_LEN):
        # print(seq)
        length = len(seq)
        if max_len > 0 and min_len is None:
            padding_needed = max_len - length
            seq = "N"*padding_needed + seq
        if min_len is not None:
            if len(seq) < min_len:
                seq = "N"*(min_len - len(seq)) + seq

            if len(seq) > min_len:
                seq = seq[(len(seq) - min_len):]
        seq = seq.lower()
        one_hot = np.array([nuc_dict[x] for x in seq]) # get stacked on top of each other
        return one_hot


    # max_len = len(max(df[self.col], key=len))



    model = load_model('./../../models/humanMedian_trainepoch.11-0.426.h5', custom_objects={'FrameSliceLayer': FrameSliceLayer})

    gpath = "./../../models/checkpoint_150000.h5"

    wgan = load_model(gpath)

    """
    Data:
    """

    tf.random.set_seed(25)
    np.random.seed(25)

    diffs = []
    init_exps = []
    opt_exps = []
    orig_vals = []

    DIM = 40
    MAX_LEN = 128
    BATCH_SIZE = args.uc
    LR = np.exp(-args.lr)

    tempnoise = tf.random.normal(shape=[BATCH_SIZE,DIM])
    selectednoise = tempnoise

    best = 10

    LOW_START = True


    if LOW_START:
    
        for i in range(1000):
            tempnoise = tf.random.normal(shape=[BATCH_SIZE,DIM])
            sequences = wgan(tempnoise)

            seqs_gen = utils.recover_seq(sequences, rev_rna_vocab)
            seqs_str = seqs_gen

            shape_ = tf.shape(np.array([encode_seq(seq) for seq in utils.recover_seq(sequences, rev_rna_vocab)]))

            seqs = tf.convert_to_tensor(np.array([encode_seq(seq) for seq in utils.recover_seq(sequences, rev_rna_vocab)]),dtype=tf.float32)

            pred =  model(seqs)

            t = tf.reshape(pred,(-1))
            t = t.numpy().astype('float')
            score = np.mean(t)

            if score < best:
                best = score
                selectednoise = tempnoise
        noise = tf.Variable(selectednoise)
    else:
        noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,DIM]))
    

    noise_small = tf.random.normal(shape=[BATCH_SIZE,DIM],stddev=1e-5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    '''
    Optimization takes place here.
    '''

    bind_scores_list = []
    bind_scores_means = []
    sequences_list = []

    means = []
    maxes = []
    iters_ = []

    OPTIMIZE = True

    DNA_SEL = False


    sequences_init = wgan(noise)

    gen_seqs_init = sequences_init.numpy().astype('float')

    seqs_gen_init = utils.recover_seq(gen_seqs_init, rev_rna_vocab)

    seqs_init = np.array([encode_seq(seq) for seq in seqs_gen_init])

    seqs_init = np.reshape(seqs_init,(-1,MAX_LEN,4))

    seqs_init = tf.convert_to_tensor(seqs_init,dtype=tf.float32)


    pred_init = model(seqs_init) 

    t = tf.reshape(pred_init,(-1))

    init_t = t.numpy().astype('float')

    init_exp = np.mean(init_t)

    max_init = np.max(init_t)

    min_init = np.min(init_t)

    if OPTIMIZE:
        iter_ = 0
        for opt_iter in tqdm(range(1300)):
            
            with tf.GradientTape() as gtape:
                gtape.watch(noise)
                sequences = wgan(noise)

                seqs_gen = utils.recover_seq(sequences, rev_rna_vocab)
                seqs_str = seqs_gen

                shape_ = tf.shape(np.array([encode_seq(seq) for seq in utils.recover_seq(sequences, rev_rna_vocab)]))

                seqs = tf.convert_to_tensor(np.array([encode_seq(seq) for seq in utils.recover_seq(sequences, rev_rna_vocab)]),dtype=tf.float32)
                
                with tf.GradientTape() as ptape:
                    ptape.watch(seqs)

                    pred =  model(seqs)
                    score = tf.reduce_mean(pred)
                    t = tf.reshape(pred,(-1))
                    mx = t.numpy().astype('float')
                    mx = np.max(mx)
                    
                    sum_ = tf.reduce_sum(t).numpy().astype('float')
                    
                    maxes.append(mx)
                    means.append(sum_/BATCH_SIZE)

                g1 = ptape.gradient(score,seqs)

                OPTIMIZE_FULL = True
                if OPTIMIZE_FULL:
                    tmp_g = g1.numpy().astype('float')
                    tmp_seqs = seqs_gen
                    tmp_lst = np.zeros(shape=(BATCH_SIZE,MAX_LEN,5))
                    for i in range(len(tmp_seqs)):
                        
                        len_ = len(tmp_seqs[i])
                        edited_g = tmp_g[i][:len_,:]
                        edited_g = np.pad(edited_g,((0,MAX_LEN-len_),(0,1)),'constant')   
                        tmp_lst[i] = edited_g   
                    
                    g1 = tf.convert_to_tensor(tmp_lst,dtype=tf.float32)

                else:
                    
                    g1 = tf.pad(g1,tf.constant([[0, 0], [0, 0], [0, 1]]),"CONSTANT")

                g1 = tf.math.scalar_mul(-1.0,g1)

                
                g2 = gtape.gradient(sequences,noise,output_gradients=g1)

            # a1 = g2 + noise_small
            a1 = g2 + noise_small
            change = [(a1,noise)]
            optimizer.apply_gradients(change)

            iters_.append(iter_)
            iter_ += 1

        sequences_opt = wgan(noise)
        
        gen_seqs_opt = sequences_opt.numpy().astype('float')

        seqs_gen_opt = utils.recover_seq(gen_seqs_opt, rev_rna_vocab)

        seqs_opt = np.array([encode_seq(seq) for seq in seqs_gen_opt])

        seqs_opt = np.reshape(seqs_opt,(-1,MAX_LEN,4))

        seqs_opt = tf.convert_to_tensor(seqs_opt,dtype=tf.float32)

        pred_opt = model(seqs_opt)

        t = tf.reshape(pred_opt,(-1))
        opt_t = t.numpy().astype('float')

        opt_exp = np.mean(opt_t)

        min_opt = np.min(opt_t)
        max_opt = np.max(opt_t)

        with open('init_mrl.npy', 'wb') as f:
            np.save(f, init_t)

        with open('opt_mrl.npy', 'wb') as f:
            np.save(f, opt_t)

        with open('seqs_mrl.npy', 'wb') as f:
            np.save(f, np.array(seqs_gen_opt))
