
from re import A, L
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import sys
sys.path.append('/home/sina/ml/gan/dev/shot0')
sys.path.insert(0, '/home/sina/ml/gan/dev/shot0/lib')
# from lib import models
# from lib import utils
import socket
import datetime
import random
import os
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '5'

from Bio import SeqIO
import pandas as pd
import numpy as np
# from BCBio import GFF
import requests, sys



BATCH_SIZE = 128
# SEQ_BATCH= 32


def fetch_seq(start, end, chr, strand):
    server = "https://rest.ensembl.org"
    
    ext = "/sequence/region/human/" + str(chr) + ":" + str(start) + ".." + str(end) + ":" + str(strand) +  "?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    return r.text

def parse_biomart(path = 'martquery_0721120207_840.txt'):

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



def fetch_seq(start, end, chr, strand):
    server = "https://rest.ensembl.org"
    
    ext = "/sequence/region/human/" + str(chr) + ":" + str(start) + ".." + str(end) + ":" + str(strand) +  "?"
    
    r = requests.get(server+ext, headers={ "Content-Type" : "text/plain"})
    
    if not r.ok:
        r.raise_for_status()
        sys.exit()

    return r.text

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
    
def select_dna_single(fname='small_seqs.npy',batch_size=64):
    refs = np.load(fname)
    indice = random.sample(range(0,refs.shape[0]),1)
    refs = refs
    return indice[0], refs    




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

def replace_seqs_coordinate(original_utr_length, original_sequence, genutrs):
    replaced = []

    for i in range(len(genutrs)):
        utr = genutrs[i]
        utr_rep = original_sequence[:7000] + utr + original_sequence[7000+original_utr_length:]
        utr_rep = utr_rep[:10500]
        if len(utr_rep) < 10500:
            utr_rep = utr_rep + (10500 - len(utr_rep)) * 'A'

        replaced.append(utr_rep)

    return replaced

def replace_xpresso_seqs_single(gens,df:pd.DataFrame,refs,index):

    if isinstance(gens, tf.Tensor):
        gens = gens.numpy().astype('str')

    seqs = []
    originals = []
    len_ = abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end']))

    origin_dna = refs[index]
    
    if len(origin_dna) != 10628:
        origin_dna = origin_dna + (10628-len(origin_dna)) * "A"
    

    for i in range(len(gens)):
        length = len(gens[i])
        diff = length - len_
        gen_dna = origin_dna[:7000] + gens[i] + origin_dna[7000+abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end'])):10500 - diff]
        
        original = origin_dna[:10500]
        
        if len(gen_dna) < 10500:
            gen_dna = gen_dna + (10500 - len(gen_dna)) * "A"

        seqs.append(gen_dna)


    seqs = tf.convert_to_tensor(seqs)

    original = tf.convert_to_tensor([original for i in range(BATCH_SIZE)])

    return seqs, original

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

logdir, checkpoint_baseline = log(samples_dir=True)

model = load_model('/home/sina/ml/gan/dev/predict/xpresso/humanMedian_trainepoch.11-0.426.h5')

model = convert_model(model)

parser = argparse.ArgumentParser()

parser.add_argument('-g', type=str, required=True)

args = parser.parse_args()

gene_name = args.g

if gene_name == 'TLR6':
    
    UTRSTARTS = [38856761,38829474,38843639]
    UTRENDS = [38856817,38829537,38843791]
    TSS = 38856817

    STRAND = -1
    CHR = 4

    UTRSTART = UTRSTARTS[0]
    UTREND = UTRENDS[0]

    # IFNG
    UTRSTART = 38856761
    UTREND = 38856817

    UTRLENGTH = UTREND - UTRSTART

elif gene_name == 'IFNG':

    UTRSTARTS = [68159616]
    UTRENDS = [68159740]
    TSS = 68159740

    STRAND = -1
    CHR = 12

    UTRSTART = UTRSTARTS[0]
    UTREND = UTRENDS[0]

    UTRLENGTH = UTREND - UTRSTART

elif gene_name == 'TNF':

    # TNF
    UTRSTARTS = [31575565]
    UTRENDS = [31575741]
    TSS = 31575565

    STRAND = 1
    CHR = 6

    UTRSTART = UTRSTARTS[0]
    UTREND = UTRENDS[0]

    UTRLENGTH = UTREND - UTRSTART

elif gene_name == 'TP53':
    
    # TNF
    UTRSTARTS = [7687377,7676595]
    UTRENDS = [7687487,7676622]
    TSS = 7687487

    STRAND = -1
    CHR = 17

    UTRSTART = UTRSTARTS[0] 
    UTREND = UTRENDS[0]

    UTRLENGTH = abs(UTREND - UTRSTART)

# original_gene_sequence = fetch_seq(start=TSS-7000,end=TSS+3500 + 128,chr=CHR,strand=STRAND)

with open(gene_name+'.txt','r') as f:
    original_gene_sequence = f.readline()

print(original_gene_sequence)
print(len(original_gene_sequence))

gpath = "/home/sina/ml/gan/dev/checkpoint_h5_old/checkpoint_50000.h5"
# gpath = "/home/sina/ml/gan/dev/gan/logs/2022.12.30-07h26m29s_neo/checkpoint_h5/checkpoint_195000.h5"
# gpath 

# seq_orig = one_hot(original_gene_sequence[:10500])

# # print(tf.shape(seqs_init))

# pred_orig = model(seq_orig) 


wgan = tf.keras.models.load_model(gpath)

"""
Data:
"""

noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,40]))

tf.random.set_seed(25)

np.random.seed(25)

diffs = []
init_exps = []

opt_exps = []

orig_vals = []

# for i in range(50):
    # print(f'Seq_n: {i}')
# noise = tf.Variable(np.random.normal(size=[2, 40]))
noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,40]))
# noise = tf.random.normal(shape=[BATCH_SIZE,40])
noise_small = tf.random.normal(shape=[BATCH_SIZE,40],stddev=1e-5)

optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)

'''
Original Gene Expression
'''

seqs_orig = one_hot([original_gene_sequence[:10500]])
pred_orig = model(seqs_orig) 
# tf.print(pred_orig)


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

seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

seqs_init = replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen_init)

print(np.shape(seqs_init))

seqs_init = one_hot(seqs_init)

print(tf.shape(seqs_init))

pred_init = model(seqs_init) 

t = tf.reshape(pred_init,(-1))

init_t = t.numpy().astype('float')
# sum_init = tf.reduce_max(t).numpy().astype('float')
# sum_init = tf.math.argmax()

if OPTIMIZE:
    
    iter_ = 0
    for opt_iter in tqdm(range(3000)):
        
        with tf.GradientTape() as gtape:

            gtape.watch(noise)
            
            sequences = wgan(noise)

            seqs_gen = recover_seq(sequences, rev_rna_vocab)

            seqs2 = replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen)
        
            seqs = one_hot(seqs2)
            seqs = tf.convert_to_tensor(seqs,dtype=tf.float32)

            # print(tf.shape(seqs))
            
            with tf.GradientTape() as ptape:

                ptape.watch(seqs)

                pred =  model(seqs)
                t = tf.reshape(pred,(-1))
                mx = np.amax(t.numpy().astype('float'),axis=0)
                mx = np.max(mx)
                
                sum_ = tf.reduce_sum(t)
                # sum_ = tf.reduce_sum(sum_).numpy().astype('float')
                maxes.append(mx)
                means.append(sum_/BATCH_SIZE)

            # tf.print(seqs)
            # print(tf.shape(seqs))
            # tf.print(pred)
            # print(tf.shape(pred))
            g1 = ptape.gradient(pred,seqs)
            # tf.print(g1)
            # print(tf.shape(g1))
            g1_ = tf.math.scalar_mul(-1.0, g1)
            
            g1 = tf.slice(g1_,[0,7000,0],[-1,128,-1])


            tmp_g = g1.numpy().astype('float')
            tmp_seqs = seqs_gen

            tmp_lst = np.zeros(shape=(BATCH_SIZE,128,5))
            for i in range(len(tmp_seqs)):
                len_ = len(tmp_seqs[i])
                
                edited_g = tmp_g[i][:len_,:]

                # print(np.reshape(edited_g[:4,:1],(-1)))
                
                # print(tf.reshape(tf.slice(g1_,[i,3500,0],[i,4,1]),(-1)).numpy().astype('float')[:4])

                edited_g = np.pad(edited_g,((0,128-len_),(0,1)),'constant')   
                
                tmp_lst[i] = edited_g
                

            # g2 = tf.reshape(g1_,(4,-1))
            # sum = tf.reduce_sum(g2,axis=1)
            # tf.print(sum)
            # f = g2.numpy()
            # for item in f:
            #     print(list(item[:-100]),sep='-')
            #     break

            # g1_ = tf.pad(g1_,paddings=[[0,0],[0,0],[0,1]])
            g1 = tf.convert_to_tensor(tmp_lst,dtype=tf.float32)

            g2 = gtape.gradient(sequences,noise,output_gradients=g1)
            # x = tf.reshape(g2,(-1))
            # sum = tf.reduce_sum(x)
            # tf.print(sum)

        a1 = g2 + noise_small
        change = [(a1,noise)]
        # a1 = a1.numpy().tolist()
        # noise = noise.numpy().tolist()
        optimizer.apply_gradients(change)

        # noise = tf.convert_to_tensor(noise,dtype=tf.float32)

        iters_.append(iter_)
        iter_ += 1

    sequences_opt = wgan(noise)

    gen_seqs_opt = sequences_opt.numpy().astype('float')

    seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)

    seqs_opt= replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen_opt)

    seqs_opt = one_hot(seqs_opt)

    pred_opt = model(seqs_opt)

    t = tf.reshape(pred_opt,(-1))
    opt_t = t.numpy().astype('float')
    # sum_opt = tf.reduce_sum(t).numpy().astype('float')

    # print(type(t_orig))
    # print(t_orig)



# gen_seqs = wgan(noise)

# gen_seqs_max = gen_seqs.numpy().astype('float')

# gen_seqs_max = gen_seqs_max[max_indices]

# seqs_gen = utils.recover_seq(gen_seqs, rev_rna_vocab)

# seqs2, origs = replace_xpresso_seqs_test(seqs_gen,df,test_refs,test_indices)

# seqs = one_hot(seqs2)

# pred =  model(seqs)

# sequences_init = sequences_init.numpy().astype('float')

# sequences_init = sequences_init[max_indices]

# seqs_gen_initial = utils.recover_seq(sequences_init, rev_rna_vocab)

# # seqs_gen = utils.recover_seq(sequences, rev_rna_vocab)

# seqs_initial, origs_test = replace_xpresso_seqs_test(seqs_gen_initial,df,test_refs,test_indices)

# seqs_initial = one_hot(seqs_initial)

# pred_initial =  model(seqs_initial)

# print('Initial Exp: {}'.format(np.average(pred)))

# print('Optimized Exp: {}'.format(np.average(pred_initial)))

with open('init_exps_'+gene_name+'.npy', 'wb') as f:
    np.save(f, init_t)

with open('opt_exps_'+gene_name+'.npy', 'wb') as f:
    np.save(f, opt_t)

with open('seqs_'+gene_name+'.npy', 'wb') as f:
    np.save(f, seqs_gen_opt)

# with open('inits.npy', 'wb') as f:
#     np.save(f, sequences_init)


# plt.scatter(iters_,maxes)
# plt.plot(iters_, maxes, '-o')
# plt.xlabel('Epoch')
# plt.ylabel('Avg. log TPM Exp.')

# plt.savefig('maxes.png')

# plt.clf()

# plt.plot(iters_, means, '-o')
# plt.xlabel('Epoch')
# plt.ylabel('Avg. log TPM Exp.')

# plt.savefig('means.png')

inits = init_t
opts = opt_t

print(np.max(inits))
print(np.min(inits))

print(np.max(opts))
print(np.min(opts))

diffs = opts - inits

print(np.mean(diffs))

positives = []
negatives = []

positive_inits = []
negative_inits = []

for i in range(len(diffs)):
    if diffs[i] >= 0:
        positive_inits.append(abs(inits[i]))
        positives.append(diffs[i])
    if diffs[i] < 0:
        negative_inits.append(abs(inits[i]))
        negatives.append(diffs[i])

print(np.mean(positives))
print(np.mean(negatives))
print(len(positives))
print(len(negatives))
print(np.mean(np.divide(positives,positive_inits)))
print(np.mean(np.divide(negatives,negative_inits)))
print(np.mean(positive_inits))
print(np.mean(negative_inits))
print(np.max(diffs))
print("Original Expression :")
print(np.power(10,pred_orig.numpy().astype('float')))
print("Max Optimized Expression :")
print(np.power(10,max(opts)))
print("Percentage Increase :")
print((max(opts)-pred_orig.numpy().astype('float'))/pred_orig.numpy().astype('float'))
