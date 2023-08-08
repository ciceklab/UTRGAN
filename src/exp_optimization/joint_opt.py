
from re import A, L
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import sys
import socket
import datetime
import random
import os
import matplotlib.pyplot as plt
from Bio import SeqIO
import pandas as pd
import numpy as np
import requests, sys
from util import *
from framepool import *
from popen import Auto_popen

import scipy.stats as stats
abs_path = '/home/sina/UTR/optimization/mrl/log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini'
Configuration = Auto_popen(abs_path)
import utils as util_motif

tf.compat.v1.enable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, required=False ,default=64)
parser.add_argument('-g', type=str, required=False ,default='IFNG')
parser.add_argument('-lr', type=int, required=False ,default=1)
parser.add_argument('-gpu', type=str, required=False ,default='-1')
parser.add_argument('-s', type=int, required=False ,default=1000)
args = parser.parse_args()

if args.gpu == '-1':
    device = 'cpu'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'
    if args.gpu.includes(','):
        device = 'cuda:1'

BATCH_SIZE = args.bs
DIM = 40
SEQ_LEN = 128
MAX_LEN = SEQ_LEN

gpath = './../../models/checkpoint_3000.h5'
tpath = './script/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth'


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

def convert_model(model_:Model):
    print(model_.summary())
    input_ = tf.keras.layers.Input(shape=( 10500, 4))
    input = input_
    for i in range(len(model_.layers)-1):

        # print(type(model_.layers[i+1]))
        
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

def gen_random_dna(len=10500,size=SEQ_LEN):
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

def select_best(scores, seqs, gc_control=False, GC=-1):
    selected_scores = []
    selected_seqs = []
    for i in range(len(scores[0])):
        best = scores[0][i]
        best_seq = seqs[0][i]
        for j in range(len(scores)):
            if scores[j][i] > best:
                if gc_control:
                    if get_gc_content(seqs[j][i]) < GC:
                        best = scores[j][i]
                        best_seq = seqs[j][i]
                else:
                    best = scores[j][i]
                    best_seq = seqs[j][i]

        selected_scores.append(best)
        selected_seqs.append(best_seq)

    return selected_seqs, selected_scores

def log(samples_dir=False):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs/", "gan_test_opt", stamp)

    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    return full_logdir, 0

if __name__ == "__main__":

    model = tf.keras.models.load_model('/mnt/sina/run/ml/gan/dev/predict/xpresso/humanMedian_trainepoch.11-0.426.h5')

    model = convert_model(model)

    gene_name = args.g

    wgan = tf.keras.models.load_model(gpath)

    """
    Data:
    """

    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,DIM]))

    tf.random.set_seed(25)

    np.random.seed(25)

    ####################### Config ############################

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

    with open(f'./genes/{gene_name}.txt','r') as f:
        original_gene_sequence = f.readline()

    seqs_orig = one_hot([original_gene_sequence[:10500]])
    pred_orig = model(seqs_orig) 

    ################ MRL/TE Optimization ######################
    mrl_model = load_framepool()
    te_model = torch.load(tpath,map_location=torch.device(device))['state_dict']  
    te_model.train().to(device)
    
    
    MODEL = "TE"
    Optimize_FrameSlice = False
    if MODEL == "TE": 
        opt_model = te_model
        Optimize_FrameSlice = False
    else:
        opt_model = mrl_model
        Optimize_FrameSlice = True

    noise = tf.Variable(tf.random.normal(shape=[BATCH_SIZE,DIM]))

    sequences_init = wgan(noise)

    gen_seqs_init = sequences_init.numpy().astype('float')

    seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

    seqs_init = replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen_init)

    seqs_init = one_hot(seqs_init)

    pred_init = model(seqs_init) 

    t = tf.reshape(pred_init,(-1))

    init_t = t.numpy().astype('float')
    
    if Optimize_FrameSlice:
        seqs_init = np.array([encode_seq_framepool(seq) for seq in seqs_gen_init])

        seqs_init = np.reshape(seqs_init,(-1,MAX_LEN,4))

        seqs = tf.convert_to_tensor(seqs_init,dtype=tf.float32)

        preds_init = opt_model(seqs)
        preds_init = preds_init.numpy().astype('float')
        pred_init = np.mean(preds_init)
        
    else:

        one_hots = one_hot_all_motif(np.array(seqs_gen_init))
        seqs = torch.tensor(one_hots,dtype=torch.double)
        seqs = torch.transpose(seqs, 1, 2)
        seqs = seqs.float().to(device)
        pred_init = opt_model.forward(seqs)
        pred_init = torch.flatten(pred_init)

        preds_init = pred_init.cpu().detach().numpy()
        pred_init = np.average(preds_init)


    
    init_mrl = np.mean(pred_init)
    max_init = np.max(pred_init)
    min_init = np.min(pred_init)
    
    predicted_mrls = []

    OPTIMIZE = True

    means = []
    maxes = []

    noise_small = tf.random.normal(shape=[BATCH_SIZE,DIM],stddev=1e-4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

    STEPS = args.s

    if OPTIMIZE:
        iter_ = 0
        for opt_iter in tqdm(range(int(STEPS))):
            
            with tf.GradientTape() as gtape:
                gtape.watch(noise)
                sequences = wgan(noise)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)
                seqs_str = seqs_gen
                
                if Optimize_FrameSlice:

                    seqs = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in recover_seq(sequences, rev_rna_vocab)]),dtype=tf.float32)
                
                else:
                    seqs = torch.tensor(np.array(one_hot_all_motif(seqs_gen),dtype=np.float32))    

                if Optimize_FrameSlice:

                    with tf.GradientTape() as ptape:
                        ptape.watch(seqs)

                        pred =  opt_model(seqs)
                        score = tf.reduce_mean(pred)
                        t = tf.reshape(pred,(-1))
                        mx = t.numpy().astype('float')
                        mx = np.max(mx)
                        
                        sum_ = tf.reduce_sum(t).numpy().astype('float')
                        
                        maxes.append(mx)
                        predicted_mrls.append(sum_/BATCH_SIZE)
                        means.append(sum_/BATCH_SIZE)

                    g1 = ptape.gradient(score,seqs)

                    OPTIMIZE_FULL = False
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

                
                else:
                    
                    seqs = torch.transpose(seqs, 1, 2)
                    seqs = seqs.float()
                    seqs = torch.tensor(seqs.to(device), requires_grad=True)
                    pred = opt_model.forward(seqs)
                    pred = torch.flatten(pred)
                    predicted_mrls.append(np.average(pred.cpu().detach().numpy()))
                    score = torch.mean(pred)
                    # predicted_mrls.append(score.cpu().detach().numpy())
                    t = torch.flatten(pred)
                    mx = t.cpu().detach().numpy()
                    mx = np.max(mx)
                    
                    sum_ = torch.mean(t).cpu().detach().numpy()
                    
                    maxes.append(mx)
                    means.append(sum_/BATCH_SIZE)
                    
                    pred.backward(torch.ones_like(pred))
                    
                    g1 = seqs.grad
                    # print(g1.grad)
                    
                    g1 = g1.cpu().detach().numpy()
                    g1 = tf.convert_to_tensor(g1)
                    # print(tf.shape(g1))
                    g1 = tf.transpose(g1, perm=[0,2,1])
                    g1 = tf.pad(g1,tf.constant([[0, 0], [0, 0], [0, 1]]),"CONSTANT")
                    g1 = tf.math.scalar_mul(-1.0,g1)
                
                
                g2 = gtape.gradient(sequences,noise,output_gradients=g1)

            a1 = g2 + noise_small
            change = [(a1,noise)]
            optimizer.apply_gradients(change)

            # iters_.append(iter_)
            iter_ += 1

        sequences_opt = wgan(noise)
        
        gen_seqs_opt = sequences_opt.numpy().astype('float')

        seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)
        
        if Optimize_FrameSlice:
            

            seqs_opt = np.array([encode_seq_framepool(seq) for seq in seqs_gen_opt])

            seqs_opt = np.reshape(seqs_opt,(-1,MAX_LEN,4))

            seqs_opt = tf.convert_to_tensor(seqs_opt,dtype=tf.float32)

            preds_opt = opt_model(seqs_opt)

            preds_opt = preds_opt.numpy().astype('float')
        
        else: 

            one_hots = np.array(one_hot_all_motif(seqs_gen_opt))
            # print(np.shape(one_hots))
            seqs = torch.tensor(one_hots,dtype=torch.double)
            seqs = torch.transpose(seqs, 1, 2)
            seqs = seqs.float().to(device)
            preds_opt = opt_model.forward(seqs)
            
            preds_opt = preds_opt.cpu().data.numpy()



        opt_mrl = np.mean(preds_opt)




    ###########################################################



    diffs = []
    init_exps = []

    opt_exps = []

    orig_vals = []


    noise_small = tf.random.normal(shape=[BATCH_SIZE,DIM],stddev=1e-5)

    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-2)

    '''
    Original Gene Expression
    '''



    '''
    Optimization takes place here.
    '''

    bind_scores_list = []
    bind_scores_means = []
    sequences_list = []



    iters_ = []

    OPTIMIZE = True

    DNA_SEL = False

    gan_noise = noise.numpy().astype('float')
    noise = tf.Variable(tf.convert_to_tensor(gan_noise,dtype=tf.float32))

    sequences_step = wgan(noise)

    gen_seqs_step = sequences_step.numpy().astype('float')

    seqs_gen_step = recover_seq(gen_seqs_step, rev_rna_vocab)

    seqs_step = replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen_step)

    seqs_step = one_hot(seqs_step)

    pred_step = model(seqs_step) 

    intermediate_pred = tf.reshape(pred_step,(-1))

    intermediate_pred = intermediate_pred.numpy().astype('float')

    seqs_mrl = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in seqs_gen_init]),dtype=tf.float32)
    seqs_te =  torch.transpose(torch.tensor(np.array(one_hot_all_motif(seqs_gen_init),dtype=np.float32)),2,1).float().to(device)

    mrl_preds_init = mrl_model(seqs_mrl).numpy().astype('float')
    te_preds_init = te_model.forward(seqs_te).cpu().data.numpy()

    means = []
    maxes = []
    
    STEPS = args.s

    seqs_collection = []
    scores_collection = []

    if OPTIMIZE:
        
        iter_ = 0
        for opt_iter in tqdm(range(STEPS)):
            
            with tf.GradientTape() as gtape:

                gtape.watch(noise)
                
                sequences = wgan(noise)
                seqs_collection.append(seqs_gen)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)

                seqs2 = replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen)
            
                seqs = one_hot(seqs2)
                seqs = tf.convert_to_tensor(seqs,dtype=tf.float32)
                
                with tf.GradientTape() as ptape:

                    ptape.watch(seqs)

                    pred =  model(seqs)
                    t = tf.reshape(pred,(-1))
                    scores_collection.append(t.numpy().astype('float'))
                    mx = np.amax(t.numpy().astype('float'),axis=0)
                    mx = np.max(mx)
                    
                    sum_ = tf.reduce_sum(t)
                    maxes.append(mx)
                    means.append(sum_/BATCH_SIZE)
                    pred = tf.math.scalar_mul(-1.0, pred)

                g1 = ptape.gradient(pred,seqs)

                
                g1 = tf.slice(g1,[0,7000,0],[-1,SEQ_LEN,-1])


                tmp_g = g1.numpy().astype('float')
                tmp_seqs = seqs_gen

                tmp_lst = np.zeros(shape=(BATCH_SIZE,SEQ_LEN,5))
                for i in range(len(tmp_seqs)):
                    len_ = len(tmp_seqs[i])
                    
                    edited_g = tmp_g[i][:len_,:]

                    edited_g = np.pad(edited_g,((0,SEQ_LEN-len_),(0,1)),'constant')   
                    
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

        seqs_opt= replace_seqs_coordinate(UTRLENGTH, original_gene_sequence, seqs_gen_opt)

        seqs_opt = one_hot(seqs_opt)

        pred_opt = model(seqs_opt)

        t = tf.reshape(pred_opt,(-1))
        opt_t = t.numpy().astype('float')


        seqs_mrl = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in seqs_gen_opt]),dtype=tf.float32)
        seqs_te =  torch.transpose(torch.tensor(np.array(one_hot_all_motif(seqs_gen_opt),dtype=np.float32)),2,1).float().to(device)

        mrl_preds_opt = mrl_model(seqs_mrl).numpy().astype('float')
        te_preds_opt = te_model.forward(seqs_te).cpu().data.numpy()

        best_seqs, best_scores = select_best(scores_collection, seqs_collection)


        with open('./outputs_joint/init_exps_'+gene_name+'.txt', 'w') as f:
            for item in init_t:
                f.write(f'{item}\n')

        with open('./outputs_joint/opt_exps_'+gene_name+'.txt', 'w') as f:
            for item in opt_t:
                f.write(f'{item}\n')

        with open('./outputs_joint/best_seqs_'+gene_name+'.txt', 'w') as f:
            for item in seqs_gen_opt:
                f.write(f'{item}\n')

        with open('./outputs_joint/init_seqs_'+gene_name+'.txt', 'w') as f:
            for item in seqs_gen_init:
                f.write(f'{item}\n')

        print("TE Optimization Step:")
        print(f"Avg. Initial TE:{np.mean(preds_init)}")
        print(f"Max Initial TE:{np.amax(preds_init)}")
        print(f"Avg. Opt TE:{np.mean(preds_opt)}")
        print(f"Max Opt TE:{np.amax(preds_opt)}")
        print(f"Avg. Exp. Before First Step: {np.mean(init_t)}")
        print("Exp. Optimizization Step:")
        print(f'Avg. Exp. After First Step: {np.mean(intermediate_pred)}')
        print(f'Avg. Best Exp. After Second Step: {np.mean(best_scores)}')
        # print("MRL:")
        # print(np.average(mrl_preds_init))
        # print(np.average(mrl_preds_opt))
        print("TE:")
        print(f'TE After First Step: {np.average(preds_opt)}')
        print(f'TE After Second Step: {np.average(te_preds_opt)}')
