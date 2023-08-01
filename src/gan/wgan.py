import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
import sys
# from IPython.display import clear_output
import time
sys.path.append('/home/sina/UTR/gan')
sys.path.insert(0, '/home/sina/UTR/gan')
from lib import models
# from.lib.models import resnet_d2, resnet_g2
from lib import utils
import socket
import datetime
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt

import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

tf.compat.v1.enable_eager_execution()


MODEL_NAME = 'WGAN-TF2'
OUTPUT_PATH = os.path.join('outputs', MODEL_NAME)
TRAIN_LOGDIR = os.path.join("logs_", "tensorflow", MODEL_NAME, 'train_data') # Sets up a log directory.
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

file_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

parser = argparse.ArgumentParser()
parser.add_argument('-d', type=str, required=False ,default='./../../data/utrs.csv')    
parser.add_argument('-bs', type=int, required=False ,default=64)
parser.add_argument('-lr', type=int, required=False ,default=5)
parser.add_argument('-mil', type=int, required=False ,default=64)
parser.add_argument('-mxl', type=int, required=False ,default=128)
parser.add_argument('-dim', type=int, required=False ,default=40)
parser.add_argument('-gpu', type=str, required=False ,default='6')
args = parser.parse_args()

def plot(x, y, logdir, name, xlabel=None, ylabel=None, title=None):

    plt.plot(x,y,'-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(logdir+'/'+name+'.png')
    plt.clf()


def plot_valid(x1, y1, x2, y2, logdir, name, xlabel=None, ylabel=None, title=None):

    plt.plot(x2,y2,'-',color='tab:blue')
    plt.plot(x1,y1,'-',color='tab:orange')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.savefig(logdir+'/'+name+'.png')
    plt.clf()    

def gradient_penalty_loss( y_true, y_pred, discriminator):
  """
  Computes gradient penalty based on prediction and weighted real / fake samples
  """
  alpha = K.random_uniform((DIM, 1, 1))
  averaged_samples = (alpha * y_pred) + ((1 - alpha) * y_true)

  gradients = K.gradients(y_pred, averaged_samples)[0]
  # compute the euclidean norm by squaring ...
  gradients_sqr = K.square(gradients)
  #   ... summing over the rows ...
  gradients_sqr_sum = K.sum(gradients_sqr,
                            axis=np.arange(1, len(gradients_sqr.shape)))
  #   ... and sqrt
  gradient_l2_norm = K.sqrt(gradients_sqr_sum)
  # compute lambda * (1 - ||grad||)^2 still for each single sample
  gradient_penalty = K.square(1 - gradient_l2_norm)
  # return the mean as loss over all the batch samples
  return K.mean(gradient_penalty)

def log(samples_dir=False,suff=None):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs/", stamp)
    if suff:
        full_logdir = full_logdir + suff
    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    
    return full_logdir, 0

data_path = './../../data/utrdb2.csv'
data_utr = pd.read_csv(data_path)
UTRdf = data_utr['seq'].to_numpy()

seqs = []

UTR_LEN = 64

for i in range(len(UTRdf)):
    if len(UTRdf[i]) < UTR_LEN+1 and len(UTRdf[i]) > int(UTR_LEN/2):
        if UTRdf[i] not in seqs:
            seqs.append(UTRdf[i])

print(len(seqs))

sequences = np.array(seqs)

sequences = [x.upper() for x in sequences]

rna_vocab = {"A":0,
             "C":1,
             "G":2,
             "U":3,
             "*":4}

rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

def one_hot_encode(seq, SEQ_LEN=UTR_LEN):
    mapping = dict(zip("ACGT*", range(5)))    
    seq2 = [mapping[i] for i in seq]
    if len(seq2) < SEQ_LEN:
        extra = [np.eye(5)[4]] * (SEQ_LEN - len(seq2))
        return np.vstack([np.eye(5)[seq2] , extra])
    return np.eye(5)[seq2]

def one_hot_encode_2(seq, SEQ_LEN=UTR_LEN):
    mapping = dict(zip("ACGT", range(4)))    
    seq2 = [mapping[i] for i in seq]
    return np.eye(4)[seq2]    

ohe_sequences = np.asarray([one_hot_encode(x) for x in sequences])


BATCH_SIZE = 64 # Batch size
ITERS = 4000 # How many iterations to train for
SEQ_LEN = UTR_LEN # Sequence length in characters
DIM = 20 # Model dimensionality.
CRITIC_ITERS = 5 # How many critic iterations per generator iteration. 
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
LR = 3e-4


LAMBDA = 10 # For gradient penalty

CURRENT_EPOCH = 1 # Epoch start from
SAVE_EVERY_N_EPOCH = 50 # Save checkpoint at every n epoch

LR = 1e-4
MIN_LR = 0.000001 # Minimum value of learning rate
DECAY_FACTOR=1.00004 # learning rate decay factor
'''
Set seed for reproducibility
'''
seed = 35
np.random.seed(seed)
# tf.set_random_seed(seed)
tf.random.set_seed(seed)

logdir, checkpoint_baseline = log(samples_dir=True)
logdir2 = ''

'''
Build GAN
'''
model_type = "resnet"
data_enc_dim = 5
data_size = SEQ_LEN * data_enc_dim
# data_size = 256
gen_layers = 3
disc_layers = 3
lmbda = 10. #lipschitz penalty hyperparameter.

SAMPLE_SIZE = 128

N_CHANNELS = DIM

G = models.resnet_g2(DIM,N_CHANNELS,SEQ_LEN,5,res_layers=gen_layers)
D = models.resnet_d2(N_CHANNELS,SEQ_LEN,5,res_layers=disc_layers)

G.summary()
D.summary()

D_optimizer = Adam(learning_rate=LR, beta_1=0.5, beta_2=0.99)
G_optimizer = Adam(learning_rate=LR, beta_1=0.5, beta_2=0.99)

EPOCHs = ITERS

@tf.function
def WGAN_GP_train_d_step(real_image, batch_size, step):

    noise = tf.random.normal([batch_size, DIM])
    epsilon = tf.random.uniform(shape=[batch_size, 1, 1], minval=0, maxval=1)
    ###################################
    # Train D
    ###################################
    with tf.GradientTape(persistent=True) as d_tape:
        with tf.GradientTape() as gp_tape:
            fake_image = G([noise], training=True)
            fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
            fake_mixed_pred = D([fake_image_mixed], training=True)
            
        # Compute gradient penalty
        grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2])) # Originally axis=[1,2]
        # grad_norms = tf.norm(grads, axis=[1,2])
        gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.))
        
        fake_pred = D([fake_image], training=True)
        real_pred = D([real_image], training=True)
        
        D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
    # Calculate the gradients for discriminator
    D_gradients = d_tape.gradient(D_loss,D.trainable_variables)
    # Apply the gradients to the optimizer
    D_optimizer.apply_gradients(zip(D_gradients,D.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)

    return D_loss, gradient_penalty

@tf.function
def WGAN_GP_train_g_step(real_image, batch_size, step):

    noise = tf.random.normal([batch_size, DIM])
    ###################################
    # Train G
    ###################################
    with tf.GradientTape() as g_tape:
        fake_image = G([noise], training=True)
        fake_pred = D([fake_image], training=True)
        G_loss = -tf.reduce_mean(fake_pred)

    G_gradients = g_tape.gradient(G_loss,
                                            G.trainable_variables)
    # Apply the gradients to the optimizer
    G_optimizer.apply_gradients(zip(G_gradients,
                                                G.trainable_variables))
    # Write loss values to tensorboard
    if step % 10 == 0:
        with file_writer.as_default():
            tf.summary.scalar('G_loss', G_loss, step=step)

    return G_loss, noise

checkpoint_path = os.path.join("checkpoints", "tensorflow", MODEL_NAME)

ckpt = tf.train.Checkpoint(generator=G,
                        discriminator=D,
                        G_optimizer=G_optimizer,
                        D_optimizer=D_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=40)

def generate_and_save_images(model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
    '''
        Generate images and plot it.
    '''
    predictions = model.predict(test_input)
    utils.save_samples(logdir, predictions, epoch, rev_rna_vocab, annotated=False)


'''
load data
'''

Train = True
validate = True

data = ohe_sequences


if validate:
    split = len(data) // 10
    # print(split)
    valid_data = data[:split]
    train_data = data[split:]
    if len(train_data) == 1: train_data = train_data[0]
    if len(valid_data) == 1: valid_data = valid_data[0]
else:
    train_data = data


train_data = train_data.astype('float32')
valid_data = valid_data.astype('float32')

train_seqs = tf.data.Dataset.from_tensor_slices(train_data).shuffle(len(train_data)).batch(BATCH_SIZE)
valid_seqs = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(len(valid_data)).batch(split)


plot_iter = 10

current_learning_rate = LR
trace = True
n_critic_count = 0

d_losses = []
g_losses = []
gradient_penalties = []
iterations = 0
iteration_numbers = []
iteration_numbers_valid = []
d_validation_losses = []

random_name = time.strftime("%Y%m%d-%H%M%S")

gen_iters = 0

if Train:
    sample_noise = tf.random.normal([BATCH_SIZE, DIM])
    generate_and_save_images(G, 0, [sample_noise], figure_size=(12,6), subplot=(3,6), save=False, is_flatten=False)

    pbar = tqdm(range(EPOCHs))
    for epoch in pbar:
        start = time.time()

        tdataset = train_seqs.enumerate()

        for step, tdata in tdataset.as_numpy_iterator():
            current_batch_size = tdata.shape[0]

            d_loss, gp = WGAN_GP_train_d_step(tdata, batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
            n_critic_count += 1
            if n_critic_count >= CRITIC_ITERS: 
                g_loss, noise = WGAN_GP_train_g_step(tdata, batch_size= tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
                gen_iters += 1
                n_critic_count = 0
            
            if step % 10 == 0:
                print ('.', end='')

        
        if epoch % SAVE_EVERY_N_EPOCH == 0 and epoch != 0:
            ckpt_save_path = ckpt_manager.save()
            utils.save_checkpoints(logdir,G,epoch)

        if epoch % 50 == 0:    
            
            generate_and_save_images(G, epoch, [sample_noise], figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False)

        os.system('clear')

        iteration_numbers.append(iterations)
        g_losses.append(-g_loss)
        d_losses.append(-d_loss)
        gradient_penalties.append(gp)
        plot(iteration_numbers, d_losses, logdir, 'discriminator_loss', xlabel="Iteration", ylabel="Discriminator Cost")
        plot(iteration_numbers, g_losses, logdir, 'generator_loss', xlabel="Iteration", ylabel="Generator Cost")
        plot(iteration_numbers, gradient_penalties, logdir, 'gradient_penalty', xlabel="Iteration", ylabel="Gradient Penalty")

        iterations+=1

        iteration_numbers_valid.append(iterations)
        fake_image_valid = G([noise], training=True)
        fake_pred_valid = D([fake_image_valid], training=True)
        real_pred_valid = D([tf.convert_to_tensor(list(valid_seqs.as_numpy_iterator())[0])], training=True)
        
        D_loss_valid = tf.reduce_mean(fake_pred_valid) - tf.reduce_mean(real_pred_valid)
        d_validation_losses.append(-D_loss_valid)

        plot_valid(iteration_numbers_valid, d_validation_losses, iteration_numbers, d_losses, logdir, 'validation_loss', xlabel="Iteration", ylabel="D Validation Loss")

    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(EPOCHs,
                                                            ckpt_save_path))


print("####################################################")
print(f"############### Gen Iterations : {gen_iters} #####")
print("####################################################")