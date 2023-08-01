import os
import operator
import pickle
import time
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import itertools
import random
import os
from polyleven import levenshtein
import operator
import pickle
import time
import tensorflow as tf

tf.random.set_seed(35)
np.random.seed(35)

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def recover_seq(samples, rev_charmap=rev_rna_vocab):
    """Convert samples to strings and save to log directory."""
    
    char_probs = samples
    argmax = np.argmax(char_probs, 2)
    seqs = []
    for line in argmax:
        s = "".join(rev_charmap[d] for d in line)
        s = s.replace('*','')
        seqs.append(s)
    return seqs 

def file_to_list(file_name,size):
    data = []
    with open(file_name,'r') as f:
        lines = f.readlines()
        
        for seq in lines:
            seq_ = seq.replace('\n','')
            data.append(seq_)
            # if len(seq) == size:
            #     data.append(seq)

    return data

def one_hot_encode(seq):
    mapping = dict(zip("ACGT", range(4)))
    seq = seq.replace('U','T')    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]

def encode(seqs):
    return np.reshape([one_hot_encode(seqs[i]) for i in range(len(seqs))],(np.array(seqs).shape[0],-1))

min_len = None
nuc_dict = {'a':[1.0,0.0,0.0,0.0],'c':[0.0,1.0,0.0,0.0],'g':[0.0,0.0,1.0,0.0], 
                'u':[0.0,0.0,0.0,1.0], 't':[0.0,0.0,0.0,1.0], 
                'n':[0.0,0.0,0.0,0.0], 'x':[1/4,1/4,1/4,1/4]}

def encode_seq_framepool(seq, max_len=128):
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

def list_to_file(filename,list):
    with open(filename + ".txt", 'w') as f:
        for element in list[:-1]:
            f.write(element+"\n")
        f.write(list[-1])       

def tsne_natural(file_name, length, key='seq'):
    df = pd.read_csv(file_name)
    seqs = np.array(df[key]).tolist()

    selected_seqs = []

    for i in range(len(seqs)):
        seq = seqs[i]
        seq = seq.upper()
        if seq not in selected_seqs and len(seq) == length:
            selected_seqs.append(seq)

    return selected_seqs

def read_real(file_name, UTR_LEN, key='seq', all= True, samples= 128 ):
    df = pd.read_csv(file_name)
    seqs = np.array(df[key]).tolist()

    selected_seqs = []

    for i in range(len(seqs)):
        if len(seqs[i]) < (UTR_LEN + 1) and len(seqs[i]) > int(UTR_LEN/2):
            seqs[i] = seqs[i].upper()
            if seqs[i] not in selected_seqs:
                selected_seqs.append(seqs[i])

    if all:
        return selected_seqs
    
    else:
        indices = []

        for i in range(len(selected_seqs)):

            indices.append(i)

        samples = np.random.choice(len(indices),samples,replace=False)

        chosen = []

        for i in range(len(samples)):
            chosen.append(selected_seqs[samples[i]])
            
        return chosen
    
def random_sample(length):
    rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "T":3}

    rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

    mapping = dict(zip([0,1,2,3],"ACGT"))

    sample = ''
    for i in range(length):
        r = random.random()
        if r < 0.6:
            sample += random.choice(['C','G'])
        else:
            sample += random.choice(['A','T'])
    # rsample = [random.randrange(4) for i in range(length)]
    # rsample = [mapping[i] for i in rsample]
    # string = ''
    # for neuc in rsample:
    #     string += neuc

    return sample 

def random_data(length, size):
    samples = []
    for i in range(size):
        samples.append(random_sample(length))

    return samples

def tsne_gen(path,length=128,BATCH_SIZE=64,DIM=40):
    wgan = tf.keras.models.load_model(path)

    selected = []
    while len(selected) < BATCH_SIZE:

        noise = tf.Variable(tf.random.normal(shape=[64,DIM]))

        sequences_init = wgan(noise)

        gen_seqs_init = sequences_init.numpy().astype('float')

        seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

        for i in range(len(seqs_gen_init)):
            if len(seqs_gen_init[i]) == length:
                selected.append(seqs_gen_init[i])

    return selected[:BATCH_SIZE]

def generate_data(path,BATCH_SIZE=64,UTR_LEN=128,DIM=40):
    wgan = tf.keras.models.load_model(path)

    gens = []

    while len(gens) < BATCH_SIZE:

        noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,DIM]))

        sequences_init = wgan(noise)

        gen_seqs_init = sequences_init.numpy().astype('float')

        seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

        for i in range(len(seqs_gen_init)):
            if len(seqs_gen_init[i]) > int(UTR_LEN/2) and len(seqs_gen_init[i])< UTR_LEN+1:
                gens.append(seqs_gen_init[i])
                if len(gens) == BATCH_SIZE:
                    break

    return gens[:BATCH_SIZE]

def gc_percentage(seq):
    count = 0.0
    for char in seq:
        if char == 'C' or char == 'G':
            count +=1

    return float(count/len(seq))

def get_gc_content(data):
    gc_content = []
    for seq in data:
        seq.replace('\n','')
        seq.replace('*','')
        gc = gc_percentage(seq)
        gc_content.append(gc)

    return gc_content

def get_gc_content_many(data):

    collection = []
    gc_contents = []
    for seq in data:
        seq = seq.upper()
        seq.replace('\n','')
        seq.replace('*','')
        gc = gc_percentage(seq)
        gc_contents.append(gc)
    
    return gc_contents

def get_4mers():
    neucs = ['A','C','G','T']
    
    mers = [p for p in itertools.product(neucs, repeat=4)]
    for i in range(len(mers)):
        mers[i] = mers[i][0] + mers[i][1] + mers[i][2] + mers[i][3]

    return mers

def get_4mer_dic(seqs):
    
    _4mers = get_4mers()
    length = 0
    dics = []

    for seq in seqs:
        dic = {}
        for item in _4mers:
            dic[item] = 0

        # Iterate With the Sliding Window
        length = len(seq)
        limit = length - 4
        for i in range(limit):
            mer = seq[i:i+4]
            dic[mer] += 1

        dics.append(dic)
    
    return dics

def euclidean_kmer(item, ref, mers):
    dist = 0
    for mer in mers:
        diff = item[mer] - ref[mer]
        dist += math.pow(diff,2)

    return dist

def euclidean_kmer_all(item,refs,mers):
    dists = []
    for ref in refs:
        dists.append(euclidean_kmer(item,ref,mers))

    return min(dists)

def hamming_dist(src, target):
    
    dists = []
    for i in range(len(src)):
        smallest = np.inf
        for j in range(len(target)):
            dist = levenshtein(src[i],target[j])
            if dist > 0 and dist < smallest:
                smallest = dist

        dists.append(dist)

    return np.array(dists)









def one_hot_motif(seq,length=128,complementary=False):
    """
    one_hot encoding on sequence
    complementary: encode nucleatide into complementary one
    """
    
    if length == -1:
        length = len(seq)
    
    # seq = str(seq)
    # setting
    seq = list(seq.replace("U","T"))
    seq_len = len(seq)
    complementary = -1 if complementary else 1
    # compose dict
    keys = ['A', 'C', 'G', 'T'][::complementary]
    oh_dict = {keys[i]:i for i in range(4)}
    # array
    oh_array = np.zeros((length,4))
    for i,C in enumerate(seq):
        try:
            oh_array[i,oh_dict[C]]=1
        except:
            continue      # for nucleotide that are not in A C G T   
    return oh_array

def one_hot_all_motif(seqs):
    length = np.max([len(seq) for seq in seqs])
    return [one_hot_motif(seqs[i], length = 128) for i in range(len(seqs))]




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