
from tqdm import tqdm
import random
random.seed(1337)
import matplotlib.pyplot as plt
import argparse
import numpy as np
np.random.seed(1337)
import pandas as pd
import torch
from .utils.framepool import *
from .utils.util import *

import random
random.seed(1337)
import os

import scipy.stats as stats

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

tf.compat.v1.enable_eager_execution()

import pandas as pd
import numpy as np
import requests, sys

def prepare_mttrans(seqs):
    seqs_init = torch.tensor(np.array(one_hot_all_motif(seqs),dtype=np.float32))

    seqs_init = torch.transpose(seqs_init, 1, 2)
    seqs_init = torch.tensor(seqs_init,dtype=torch.float32).to('cuda:1')
    return seqs_init

def prepare_framepool(seqs):
    return tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in seqs]),dtype=tf.float32)

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, required=False ,default='./../../data/utrs.csv') 
parser.add_argument('-uc', type=int, required=False ,default=64)
parser.add_argument('-lr', type=int, required=False ,default=5)
parser.add_argument('-gpu', type=str, required=False ,default='-1')
parser.add_argument('-s', type=str, required=False ,default=2000)
args = parser.parse_args()

# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

BATCH_SIZE = 64
DIM = 40
SEQ_LEN = 128
UTR_LEN = 128
gpath = './../../models/checkpoint_3000.h5'
mrl_path = './../../models/utr_model_combined_residual_new.h5'

path = './script/checkpoint/RL_hard_share_MTL/3R/schedule_MTL-model_best_cv1.pth'
val_path = './script/checkpoint/RL_hard_share_MTL/3M/schedule_lr-model_best_cv1.pth'


TASK = 'MMRL'
OPT = 'FMRL'
VAL = "MMRL"

if VAL == 'FMRL':
    MT_VALID = False
else:
    MT_VALID = True

def select_best(scores, seqs):
    selected_scores = []
    selected_seqs = []
    for i in range(len(scores[0])):
        best = scores[0][i]
        best_seq = seqs[0][i]
        for j in range(len(scores)-1):
            if scores[j+1][i] > best:
                best = scores[j+1][i]
                best_seq = seqs[j+1][i]
        selected_scores.append(best)
        selected_seqs.append(best_seq)

    return selected_seqs, selected_scores

if __name__ == '__main__':
    
    if OPT == 'FMRL':
        Optimize_FrameSlice = True
    else:
        Optimize_FrameSlice = False

    if Optimize_FrameSlice:
        model = load_framepool(mrl_path)
        validation_model = torch.load(val_path,map_location=torch.device('cuda:1'))['state_dict']      
        validation_model.train()

    else:

        if MT_VALID:
            validation_model = torch.load(val_path,map_location=torch.device('cuda:1'))['state_dict']  
            validation_model.train()   

            model = torch.load(path,map_location=torch.device('cuda:1'))['state_dict']  
            model.train()   
        else:
            validation_model = load_framepool(mrl_path)

            model = torch.load(path,map_location=torch.device('cuda:1'))['state_dict']  
            model.train()      



    wgan = tf.keras.models.load_model(gpath)

    """
    Data:
    """

    tf.random.set_seed(265)
    np.random.seed(4354)

    diffs = []
    init_exps = []
    opt_exps = []
    orig_vals = []

    DIM = 40
    MAX_LEN = 128
    # BATCH_SIZE = args.uc
    LR = np.exp(-args.lr)

    tempnoise = tf.random.normal(shape=[BATCH_SIZE,DIM])
    selectednoise = tempnoise

    best = 10

    LOW_START = False


    if LOW_START:
    
        for i in range(10000):
            tempnoise = tf.random.normal(shape=[BATCH_SIZE,DIM])
            sequences = wgan(tempnoise)

            seqs_gen = recover_seq(sequences, rev_rna_vocab)
            seqs_str = seqs_gen

            shape_ = tf.shape(np.array([encode_seq_framepool(seq) for seq in recover_seq(sequences, rev_rna_vocab)]))

            seqs = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in recover_seq(sequences, rev_rna_vocab)]),dtype=tf.float32)

            
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
    

    noise_small = tf.random.normal(shape=[BATCH_SIZE,DIM],stddev=1e-4)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-1)

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

    init_pos, init_neg = motif_count(seqs_gen_init)
    
    if Optimize_FrameSlice:
        seqs = prepare_framepool(seqs_gen_init)

        seqs_init = prepare_mttrans(seqs_gen_init)
        # seqs_init = torch.tensor(seqs_init, requires_grad=True)
        pred_init = validation_model.forward(seqs_init)
        pred_init = torch.flatten(pred_init)

        preds_init = pred_init.cpu().detach().numpy()

        val_pred_init = np.mean(preds_init)

        pred_init = model(seqs)
        
    else:


        one_hots = one_hot_all_motif(np.array(seqs_gen_init))
        seqs = torch.tensor(one_hots,dtype=torch.double)
        seqs = torch.transpose(seqs, 1, 2)
        seqs = seqs.float().to('cuda:1')

        if MT_VALID:

            val_pred = validation_model.forward(seqs)
            pred_init = torch.flatten(val_pred)

            preds_init = pred_init.cpu().detach().numpy()

            val_pred_init = np.mean(preds_init)

        else:

            seqs_init = prepare_framepool(seqs_gen_init)

            val_pred = validation_model(seqs_init)

            score = tf.reduce_mean(val_pred)
            t = tf.reshape(val_pred,(-1))

            sum_ = tf.reduce_sum(t).numpy().astype('float')

            val_pred_init = sum_/BATCH_SIZE


        pred_init = model.forward(seqs)
    
    if Optimize_FrameSlice:

        t = tf.reshape(pred_init,(-1))

        init_t = t.numpy().astype('float')
        
    else:
        
        t = torch.flatten(pred_init)
        t.float()
        
        init_t = t.cpu().detach().numpy()

    # print(init_t)

    init_exp = np.mean(init_t)

    max_init = np.max(init_t)

    min_init = np.min(init_t)
    
    predicted_mrls = []

    STEPS = 3000

    seqs_collection = []
    scores_collection = []
    if OPTIMIZE:
        iter_ = 0
        for opt_iter in tqdm(range(int(STEPS))):
            
            with tf.GradientTape() as gtape:
                gtape.watch(noise)
                sequences = wgan(noise)

                seqs_gen = recover_seq(sequences, rev_rna_vocab)
                seqs_collection.append(seqs_gen)
                seqs_str = seqs_gen
                
                if Optimize_FrameSlice:

                    seqs = tf.convert_to_tensor(np.array([encode_seq_framepool(seq) for seq in recover_seq(sequences, rev_rna_vocab)]),dtype=tf.float32)
                
                else:
                    seqs = torch.tensor(np.array(one_hot_all_motif(seqs_gen),dtype=np.float32))    

                if Optimize_FrameSlice:

                    with tf.GradientTape() as ptape:
                        ptape.watch(seqs)

                        pred =  model(seqs)
                        score = tf.reduce_mean(pred)
                        t = tf.reshape(pred,(-1))
                        mx = t.numpy().astype('float')
                        scores_collection.append(mx)
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
                    seqs = torch.tensor(seqs.to('cuda:1'), requires_grad=True)
                    pred = model(seqs)
                    pred = torch.flatten(pred)
                    predicted_mrls.append(np.average(pred.cpu().detach().numpy()))
                    scores_collection.append(pred.cpu().detach().numpy())
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
                    
                    g1 = g1.cpu().detach().numpy()
                    g1 = tf.convert_to_tensor(g1)
                    g1 = tf.transpose(g1, perm=[0,2,1])
                    g1 = tf.pad(g1,tf.constant([[0, 0], [0, 0], [0, 1]]),"CONSTANT")
                    g1 = tf.math.scalar_mul(-1.0,g1)
                
                
                g2 = gtape.gradient(sequences,noise,output_gradients=g1)

            a1 = g2 + noise_small
            change = [(a1,noise)]
            optimizer.apply_gradients(change)

            iters_.append(iter_)
            iter_ += 1

            # print("SEQ")

        best_seqs, best_scores = select_best(scores_collection, seqs_collection)
        # print(best_seqs)

        sequences_opt = wgan(noise)
        
        gen_seqs_opt = sequences_opt.numpy().astype('float')

        seqs_gen_opt = recover_seq(gen_seqs_opt, rev_rna_vocab)

        opt_pos, opt_neg = motif_count(seqs_gen_opt)
        
        if Optimize_FrameSlice:
            
            seqs_opt = prepare_framepool(seqs_gen_opt)

            seqs_opt_v = prepare_mttrans(best_seqs)
            # seqs_init = torch.tensor(seqs_init, requires_grad=True)
            pred_opt_v = validation_model.forward(seqs_opt_v)
            pred_opt_v = torch.flatten(pred_opt_v)

            preds_opt_v = pred_opt_v.cpu().detach().numpy()

            val_pred_opt = np.mean(preds_opt_v)
        
        else: 

            one_hots = np.array(one_hot_all_motif(seqs_gen_opt))
            # print(np.shape(one_hots))
            seqs = torch.tensor(one_hots,dtype=torch.double)
            seqs = torch.transpose(seqs, 1, 2)
            seqs = seqs.float().to('cuda:1')

            if MT_VALID:
            
                val_pred = validation_model.forward(prepare_mttrans(best_seqs))
                pred_opt = torch.flatten(val_pred)

                preds_opt = pred_opt.cpu().detach().numpy()

                val_pred_opt = np.mean(preds_opt)

            else:

                seqs_init = np.array([encode_seq_framepool(seq) for seq in best_seqs])

                seqs_init = np.reshape(seqs_init,(-1,MAX_LEN,4))

                seqs_init = tf.convert_to_tensor(seqs_init,dtype=tf.float32)

                val_pred = validation_model(seqs_init)

                score = tf.reduce_mean(val_pred)
                t = tf.reshape(val_pred,(-1))

                sum_ = tf.reduce_sum(t).numpy().astype('float')

                # pred(sum_)
                
                val_pred_opt = sum_/BATCH_SIZE


        pred_opt = model(seqs)
        
        if Optimize_FrameSlice:

            t = tf.reshape(pred_opt,(-1))
            
            opt_t = t.numpy().astype('float')
            
        else:
            
            t = torch.flatten(pred_opt)
        
        
            opt_t = t.cpu().detach().numpy()

        opt_exp = np.mean(opt_t)

        min_opt = np.min(opt_t)
        max_opt = np.max(opt_t)

        with open(f'init_mrl_{OPT}_{VAL}_{STEPS}.txt', 'w') as f:
            f.writelines([str(x)+'\n' for x in init_t])

        with open(f'opt_mrl_{OPT}_{VAL}_{STEPS}.txt', 'w') as f:
            f.writelines([str(x)+'\n' for x in best_scores])

        with open(f'opt_seqs_{OPT}_{VAL}_{STEPS}.txt', 'w') as f:
            f.writelines([str(x)+'\n' for x in best_seqs])

        with open(f'init_seqs_{OPT}_{VAL}_{STEPS}.txt', 'w') as f:
            f.writelines([str(x)+'\n' for x in seqs_gen_init])
    
        print(val_pred_init)
        print(val_pred_opt)
        print(np.average(init_t))
        print(np.max(init_t))
        print(np.average(opt_t))
        print(np.max(opt_t))
        print(f"best seqs average MRL: {np.average(best_scores)}")
        x = [i for i in range(int(STEPS))]
        plt.plot(x, predicted_mrls)
        plt.savefig('opts')

        print(f'Motifs: \n init pos: {init_pos} - init neg: {init_neg} \n opt pos: {opt_pos} - opt neg: {opt_neg}')