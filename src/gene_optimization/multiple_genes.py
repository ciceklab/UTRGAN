from re import A, L
import numpy as np
import pandas as pd
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import sys
sys.path.append('/home/sina/ml/gan/dev/shot0')
sys.path.insert(0, '/home/sina/ml/gan/dev/shot0/lib')
import argparse
import socket
import datetime
import random
import os
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

from Bio import SeqIO
import pandas as pd
import numpy as np
# from BCBio import GFF
import requests, sys

np.random.seed(25)

parser = argparse.ArgumentParser()

parser.add_argument('-gp', type=str, required=True, default='./../../data/gene_info.txt')
parser.add_argument('-seqs', type=str, required=True,default='./../../data/seqs.npy')
parser.add_argument('-dc', type=str, required=False, default=32)
parser.add_argument('-uc', type=int, required=False ,default=32)
parser.add_argument('-lr', type=int, required=False ,default=5)
parser.add_argument('-gpu', type=str, required=False ,default='-1')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

BATCH_SIZE = int(args.dc)
SEQ_BATCH= int(args.uc)
LR = np.exp(-int(args.lr))

def fetch_seq(start, end, chr, strand):
    server = "https://rest.ensembl.org"
    
    ext = "/sequence/region/human/" + str(chr) + ":" + str(start) + ".." + str(end) + ":" + str(strand) +  "?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    return r.text

def parse_biomart(path = 'reference_gene_info.txt'):

    file = path

    fasta_sequences = SeqIO.parse(open(file),'fasta')

    genes = []
    ustarts = []
    uends = []
    seqs = []
    strands = []
    tsss = []
    chromosomes = []

    counter = 0

    for fasta in fasta_sequences:

        name, sequence = fasta.id, str(fasta.seq)
        
        if sequence != "Sequenceunavailable":

            counter += 1

            listed = name.split('|')
            # print(len(listed))

            if len(listed) == 8:

                genes.append(listed[1])
                chromosomes.append(listed[2])
                
                if ';' in listed[3]:
                    ustart = listed[3].split(';')[0]
                    uend = listed[4].split(';')[0]
                else:
                    ustart = listed[3]
                    uend = listed[4]

                strand = int(listed[0])

                if strand == -1:
                    ustarts.append(ustart)
                    uends.append(uend)
                else:
                    ustarts.append(uend)
                    uends.append(ustart)


                strands.append(str(strand))
                
                tsss.append(listed[-1])
                
                seqs.append(sequence)

    df = pd.DataFrame({'utr':seqs,'gene':genes, 'chr': chromosomes,'utr_start':ustarts,'utr_end':uends,'tss':tsss,'strand':strands})
    return df


def convert_model(model_:Model):
    print(model_.summary())
    input_ = tf.keras.layers.Input(shape=( 10500, 4))
    input = input_
    for i in range(len(model_.layers)-1):

        print(type(model_.layers[i+1]))
        
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

    model = tf.keras.Model(inputs=input_, outputs=output)
    model.compile(loss="mse", optimizer="adam")
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

def gen_random_dna(len=10500,size=64):
    list_ = ['A','C','G','T']
    dnas = []
    for i in range(size):
            
        list_ = ['A','C','G','T']
        mydna = 'AGT'
        for i in range(len-3):
            char = list_[random.randint(0,3)]
            mydna = mydna + char
        dnas.append(mydna)


    return dnas

def read_genes(file_name):
    with open(file_name) as f:
        lines = f.readlines()

    lines = [lines[i].replace('\n','') for i in range(len(lines))]

    return np.array(lines)
    
def select_dna_single(fname='./../../data/genes.txt',batch_size=64):
    refs = read_genes(fname)
    indice = random.sample(range(0,refs.shape[0]),1)
    refs = refs
    return indice[0], refs    

def select_dna_batch(fname='./../../data/genes.txt',batch_size=64,dna_batch=32):
    refs = read_genes(fname)
    indice = random.sample(range(0,refs.shape[0]),batch_size*dna_batch)
    refs = refs
    return indice, refs


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


def replace_xpresso_seqs_single_v2(gens,df:pd.DataFrame,refs,indices,batch_size=SEQ_BATCH):
    seqs = []
    # originals = []
    ref_len = len(indices)  

    for i in range(len(gens)):
        length = len(gens[i])        
        for j in range(SEQ_BATCH):
            index = indices[i*SEQ_BATCH+j]            
            len_ = abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end']))

            diff = length - len_
            origin_dna = refs[i*SEQ_BATCH+j]            
            if len(origin_dna) != 10628:
                origin_dna = origin_dna + (10628-len(origin_dna)) * "A"
                    
            gen_dna = origin_dna[:7000] + gens[i] + origin_dna[7000+abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end'])):10500 - diff]
            if len(gen_dna) != 10500:
                gen_dna = gen_dna + (10500-len(gen_dna)) * "A"

            seqs.append(gen_dna)

    seqs = tf.convert_to_tensor(seqs)
    return seqs, []


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

    df = parse_biomart(args.gp)

    df = df[:8200]

    model = load_model('/home/sina/ml/gan/dev/predict/xpresso/humanMedian_trainepoch.11-0.426.h5')

    model = convert_model(model)


    wgan = tf.keras.models.load_model('./../../models/checkpoint_50000.h5')

    """
    Data:
    """

    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,40]))

    tf.random.set_seed(25)



    diffs = []
    init_exps = []

    opt_exps = []

    orig_vals = []

    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,40]))
    noise_small = tf.random.normal(shape=[BATCH_SIZE,40],stddev=1e-5)

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
    indices, refs = select_dna_batch(fname=args.seqs,batch_size=BATCH_SIZE,dna_batch=SEQ_BATCH)

    sequences_init = wgan(noise)

    gen_seqs_init = sequences_init.numpy().astype('float')

    seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

    seqs_init, origs = replace_xpresso_seqs_single_v2(seqs_gen_init,df,refs,indices)

    seqs_init = one_hot(seqs_init)

    print(tf.shape(seqs_init))

    pred_init = model(seqs_init) 

    t = tf.reshape(pred_init,(-1))

    init_t = t.numpy().astype('float')

    if OPTIMIZE:

        indices, refs = select_dna_batch(fname='./../../data/genes.txt',batch_size=BATCH_SIZE,dna_batch=SEQ_BATCH)
        
        iter_ = 0
        for opt_iter in tqdm(range(3000)):
            
            with tf.GradientTape() as gtape:
                gtape.watch(noise)
                
                sequences = wgan(noise)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)

                seqs2, origs = replace_xpresso_seqs_single_v2(seqs_gen,df,refs,indices)
            
                seqs = one_hot(seqs2)
                
                with tf.GradientTape() as ptape:
                    ptape.watch(seqs)

                    pred =  model(seqs)
                    t = tf.reshape(pred,(SEQ_BATCH,-1))
                    mx = np.amax(t.numpy().astype('float'),axis=0)
                    mx = np.max(mx)
                    
                    sum_ = tf.reduce_sum(t,axis=0)
                    sum_ = tf.reduce_sum(sum_).numpy().astype('float')
                    maxes.append(mx)
                    means.append(sum_/BATCH_SIZE)

                g1 = ptape.gradient(pred,seqs)
                g1 = tf.math.scalar_mul(-1.0, g1)
                g1 = tf.slice(g1,[0,7000,0],[-1,128,-1])

                tmp_g = g1.numpy().astype('float')
                tmp_seqs = seqs_gen

                tmp_lst = np.zeros(shape=(BATCH_SIZE,128,5))
                for i in range(len(tmp_seqs)):
                    len_ = len(tmp_seqs[i])
                    edited_g = tmp_g[i][:len_,:]
                    edited_g = np.pad(edited_g,((0,128-len_),(0,1)),'constant')   
                    tmp_lst[i] = edited_g
                    
                g1 = tf.convert_to_tensor(tmp_lst,dtype=tf.float32)

                g2 = gtape.gradient(sequences,noise,output_gradients=g1)


            a1 = g2 + noise_small
            change = [(a1,noise)]

            optimizer.apply_gradients(change)

            iters_.append(iter_)
            iter_ += 1

        sequences_opt = wgan(noise)

        gen_seqs_opt = sequences_opt.numpy().astype('float')

        seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)

        seqs_opt, origs = replace_xpresso_seqs_single_v2(seqs_gen_opt,df,refs,indices)

        seqs_opt = one_hot(seqs_opt)

        pred_opt = model(seqs_opt)

        t = tf.reshape(pred_opt,(-1))
        opt_t = t.numpy().astype('float')


    with open('single_init_exps_max.npy', 'wb') as f:
        np.save(f, init_t)

    with open('single_opt_exps_max.npy', 'wb') as f:
        np.save(f, opt_t)

    with open('single_opt_exp_seqs.npy', 'wb') as f:
        np.save(f, seqs_gen_opt)


