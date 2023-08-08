from re import A, L
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
import sys
sys.path.append('/home/sina/ml/gan/dev/shot0')
sys.path.insert(0, '/home/sina/ml/gan/dev/shot0/lib')
import argparse
from util import *
from framepool import *

tf.compat.v1.enable_eager_execution()

from Bio import SeqIO
import pandas as pd
import numpy as np
import requests, sys
from popen import Auto_popen

abs_path = '/home/sina/UTR/optimization/mrl/log/Backbone/RL_hard_share/3M/small_repective_filed_strides1113.ini'
Configuration = Auto_popen(abs_path)

np.random.seed(25)

parser = argparse.ArgumentParser()
parser.add_argument('-bs', type=int, required=False ,default=64)
parser.add_argument('-dc', type=int, required=False ,default=8)
parser.add_argument('-lr', type=int, required=False ,default=1)
parser.add_argument('-gpu', type=str, required=False ,default='-1')
parser.add_argument('-s', type=int, required=False ,default=3000)
args = parser.parse_args()

if args.gpu == '-1':
    device = 'cpu'
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda'
    if args.gpu.includes(','):
        device = 'cuda:1'

BATCH_SIZE = args.bs
SEQ_BATCH = args.dc
UTR_LEN = 128
DIM = 40
gpath = './../../models/checkpoint_3000.h5'
mrl_path = './../../models/utr_model_combined_residual_new.h5'
exp_path = './../../models/humanMedian_trainepoch.11-0.426.h5'
tpath = './script/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth'
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
    # print(model_.summary())
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

def select_dna_batch(fname='./../../data/genes.txt',dna_batch=32):
    refs = read_genes(fname)
    indices = random.sample(range(0,refs.shape[0]),dna_batch)
    refs = refs
    gene_names = []
    for i in range(len(indices)):
        gene_names.append(df.iloc[indices[i]]['gene'])
    return indices, refs, gene_names

def replace_xpresso_seqs(gens,df:pd.DataFrame,refs,index):
    seqs = []

    for i in range(len(gens)):
        length = len(gens[i])        
      
        len_ = abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end']))

        diff = length - len_
        origin_dna = refs[index]            
        if len(origin_dna) != 10628:
            origin_dna = origin_dna + (10628-len(origin_dna)) * "A"
                
        gen_dna = origin_dna[:7000] + gens[i] + origin_dna[7000+abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end'])):10500 - diff]
        if len(gen_dna) != 10500:
            gen_dna = gen_dna + (10500-len(gen_dna)) * "A"

        seqs.append(gen_dna)

    seqs = tf.convert_to_tensor(seqs)
    return seqs, []

def replace_xpresso_seqs_single_v2(gens,df:pd.DataFrame,refs,indices,batch_size=SEQ_BATCH):
    seqs = []
    # originals = []
    ref_len = len(indices)  

    for i in range(len(gens)):
        length = len(gens[i])        
        for j in range(len(indices)):
            index = indices[j]            
            len_ = abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end']))

            diff = length - len_
            origin_dna = refs[j]            
            if len(origin_dna) != 10628:
                origin_dna = origin_dna + (10628-len(origin_dna)) * "A"
                    
            gen_dna = origin_dna[:7000] + gens[i] + origin_dna[7000+abs(int(df.iloc[index]['utr_start']) - int(df.iloc[index]['utr_end'])):10500 - diff]
            if len(gen_dna) != 10500:
                gen_dna = gen_dna + (10500-len(gen_dna)) * "A"

            seqs.append(gen_dna)

    seqs = tf.convert_to_tensor(seqs)
    return seqs, []

def select_best(scores, seqs, gc_control=False, GC=-1):
    selected_scores = []
    selected_seqs = []
    for i in range(len(scores[0])):
        best = scores[1][i]
        best_seq = seqs[1][i]
        for j in range(len(scores)-1):
            if scores[j+1][i] > best:
                if gc_control:
                    if get_gc_content(seqs[j][i]) < GC:
                        best = scores[j+1][i]
                        best_seq = seqs[j+1][i]
                else:
                    best = scores[j+1][i]
                    best_seq = seqs[j+1][i]

        selected_scores.append(best)
        selected_seqs.append(best_seq)

    return selected_seqs, selected_scores


if __name__ == '__main__':

    df = parse_biomart('./../../data/gene_info.txt')

    # Genes retreived from Ensemble
    df = df[:8200]

    model = tf.keras.models.load_model(exp_path)

    model = convert_model(model)

    wgan = tf.keras.models.load_model(gpath)

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
    indices, refs, names = select_dna_batch(dna_batch=SEQ_BATCH)

    sequences_init = wgan(noise)

    gen_seqs_init = sequences_init.numpy().astype('float')

    seqs_gen_init = recover_seq(gen_seqs_init, rev_rna_vocab)

    seqs_init, origs = replace_xpresso_seqs_single_v2(seqs_gen_init,df,refs,indices)

    seqs_init = one_hot(seqs_init)


    pred_init = model(seqs_init) 

    pred_init = tf.reshape(pred_init,(SEQ_BATCH,-1))

    init_t = tf.reduce_mean(pred_init,axis=0)

    init_t = init_t.numpy().astype('float')

    mrl_model = load_framepool()
    te_model = torch.load(tpath,map_location=torch.device(device))['state_dict']      
    te_model.train().to(device)



    ########### MRL and TE check before optimization ###################

    seqs_mrl = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in seqs_gen_init]),dtype=tf.float32)
    seqs_te =  torch.transpose(torch.tensor(np.array(one_hot_all_motif(seqs_gen_init),dtype=np.float32)),2,1).float().to(device)

    mrl_preds_init = mrl_model(seqs_mrl).numpy().astype('float')
    te_preds_init = te_model.forward(seqs_te).cpu().data.numpy()

    ####################################################################

    STEPS = args.s

    seqs_collection = []
    scores_collection = []
    if OPTIMIZE:

        
        iter_ = 0
        for opt_iter in tqdm(range(STEPS)):
            
            with tf.GradientTape() as gtape:
                gtape.watch(noise)
                
                sequences = wgan(noise)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)
                seqs_collection.append(seqs_gen)

                g1_ = tf.zeros_like(sequences)

                scores_collection_temp = []
                means_temp = []
                maxes_temp = []


                for indx in range(len(indices)):

                    seqs2, origs = replace_xpresso_seqs(seqs_gen,df,refs,indices[indx])
                
                    seqs = one_hot(seqs2)
                    
                    with tf.GradientTape() as ptape:
                        ptape.watch(seqs)

                        pred =  model(seqs)
                        t = tf.reshape(pred,(-1))
                        mx = np.amax(t.numpy().astype('float'),axis=0)
                        mx = np.max(mx)
                        

                        scores_collection_temp.append(t.numpy().astype('float'))
                        nt = t.numpy().astype('float')
                        maxes_temp.append(mx)
                        means_temp.append(np.sum(t)/BATCH_SIZE)

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

                    g1_ = tf.math.add(g1, g1_)

                scores_collection.append(np.mean(scores_collection_temp,axis=0)/BATCH_SIZE)
                means.append(np.mean(means_temp))
                maxes.append(np.max(maxes_temp))

                g2 = gtape.gradient(sequences,noise,output_gradients=g1_)


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

        pred_opt = tf.reshape(pred_opt,(SEQ_BATCH,-1))


        t = tf.reduce_mean(pred_opt,axis=0)
        opt_t = t.numpy().astype('float')

    ########### MRL and TE check after optimization ####################

    seqs_mrl = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in seqs_gen_opt]),dtype=tf.float32)
    seqs_te =  torch.transpose(torch.tensor(np.array(one_hot_all_motif(seqs_gen_opt),dtype=np.float32)),2,1).float().to(device)

    mrl_preds_opt = mrl_model(seqs_mrl).numpy().astype('float')
    te_preds_opt = te_model.forward(seqs_te).cpu().data.numpy()

    ####################################################################

    best_seqs, best_scores = select_best(scores_collection, seqs_collection)

    with open('./outputs/mul_init_exps.txt', 'w') as f:
        for item in init_t:
            f.write(f'{item}\n')

    with open('./outputs/mul_best_exps.txt', 'w') as f:
        for item in best_scores:
            f.write(f'{item}\n')

    with open('./outputs/mul_opt_exps.txt', 'w') as f:
        for item in opt_t:
            f.write(f'{item}\n')

    with open('./outputs/mul_best_seqs.txt', 'w') as f:
        for item in best_seqs:
            f.write(f'{item}\n')

    with open('./outputs/mul_init_seqs.txt', 'w') as f:
        for item in seqs_gen_init:
            f.write(f'{item}\n')



    print("Genes:")
    print(names)
    print(f"Average Initial Expression: {np.average(init_t)}")
    print(f"Best Expression: {np.average(best_scores)}")


