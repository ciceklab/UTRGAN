import numpy as np
import pandas as pd
import pdb
import tensorflow as tf
import sys
from IPython.display import clear_output
import time
sys.path.append('/home/sina/ml/gan/dev/gan')
sys.path.insert(0, '/home/sina/ml/gan/dev/gan')
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

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

MODEL_NAME = 'WGAN-TF2'
OUTPUT_PATH = os.path.join('outputs', MODEL_NAME)
TRAIN_LOGDIR = os.path.join("logs_", "tensorflow", MODEL_NAME, 'train_data') # Sets up a log directory.
if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


# gpus = tf.config.list_physical_devices('GPU')
# # print(gpus)
# if gpus:
# #   print("GPU MANAGER : #########################################################################")
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.set_visible_devices(gpus[4], 'GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

file_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)


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
  alpha = K.random_uniform((32, 1, 1))
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

def log(samples_dir=False):
    stamp = datetime.date.strftime(datetime.datetime.now(), "%Y.%m.%d-%Hh%Mm%Ss") + "_{}".format(socket.gethostname())
    full_logdir = os.path.join("./logs/", stamp)
    os.makedirs(full_logdir, exist_ok=True)
    if samples_dir: os.makedirs(os.path.join(full_logdir, "samples"), exist_ok=True)
    log_dir = "{}:{}".format(socket.gethostname(), full_logdir)
    return full_logdir, 0

data_path = "/home/sina/ml/gan/dev/gan/cleaned.csv"
# data_path = "/home/sina/ml/gan/dev/all_utr.csv"
data_utr = pd.read_csv(data_path)
UTRdf = data_utr['seq'].to_numpy()
# UTRlen = data_utr["length"].to_list()
# max_length = max(UTRlen)
# UTRdf = UTRdf.values

# data_path_mam = "/home/sina/ml/gan/data/mam_utr.csv"
# data_utr_mam = pd.read_csv(data_path_mam)
# UTRmam = data_utr_mam['seq'].to_numpy()

# data_path_rod = "/home/sina/ml/gan/data/rod_utr.csv"
# data_utr_rod = pd.read_csv(data_path_rod)
# UTRrod = data_utr_rod['seq'].to_numpy()

seqs = []

# for i in range(len(UTRrod)):
#   if 'W' not in UTRrod[i] and 'Y' not in UTRrod[i] and 'R' not in UTRrod[i] and 'D' not in UTRrod[i] and 'K' not in UTRrod[i] and 'N' not in UTRrod[i] and len(UTRrod[i]) < 201 and len(UTRrod[i]) > 120:
#     seqs.append(UTRrod[i])

# for i in range(len(UTRmam)):
#   if 'S' not in UTRmam[i] and 'Q' not in UTRmam[i] and 'M' not in UTRmam[i] and 'E' not in UTRmam[i] and 'B' not in UTRmam[i] and 'V' not in UTRmam[i] and 'W' not in UTRmam[i] and 'Y' not in UTRmam[i] and 'R' not in UTRmam[i] and 'D' not in UTRmam[i] and 'K' not in UTRmam[i] and 'N' not in UTRmam[i] and len(UTRmam[i]) < 201 and len(UTRmam[i]) > 120:
#     seqs.append(UTRmam[i])

UTR_LEN = 128



for i in range(len(UTRdf)):
    dna = set('ACTG')
    if len(UTRdf[i]) < UTR_LEN+1 and len(UTRdf[i]) > 63:
        # if UTRdf[i][:4] not in ["AAAA","CCCC","GGGG","TTTT"] and 'N' not in UTRdf[i] and 'R' not in UTRdf[i] and 'K' not in UTRdf[i] and 'W' not in UTRdf[i] and 'Y' not in UTRdf[i] and 'M' not in UTRdf[i]:
        # if UTRdf[i][:4] not in ["AAAA","CCCC","GGGG","TTTT"] and all(base.upper() in dna for base in UTRdf[i]):
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
ITERS = 200000 # How many iterations to train for
SEQ_LEN = UTR_LEN # Sequence length in characters
DIM = 32 # Model dimensionality.
CRITIC_ITERS = 5 # How many critic iterations per generator iteration. 
LAMBDA = 10 # Gradient penalty lambda hyperparameter.
MAX_N_EXAMPLES = 10000000 # Max number of data examples to load.
LR = 1e-3


LAMBDA = 10 # For gradient penalty

CURRENT_EPOCH = 1 # Epoch start from
SAVE_EVERY_N_EPOCH = 2000 # Save checkpoint at every n epoch

LR = 1e-5
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
generator_layers = 10
disc_layers = 7
lmbda = 10. #lipschitz penalty hyperparameter.

SAMPLE_SIZE = 2000

N_CHANNELS = 32

G = models.resnet_g2(DIM,N_CHANNELS,SEQ_LEN,5,res_layers=5)
D = models.resnet_d2(N_CHANNELS,SEQ_LEN,5,res_layers=5)

G.summary()
D.summary()

# D_optimizer = Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)
# G_optimizer = Adam(learning_rate=LR, beta_1=0.5, beta_2=0.9)

# EPOCHs = 30

# @tf.function
# def WGAN_GP_train_d_step(real_image, batch_size, step):
#     '''
#         One discriminator training step
        
#         Reference: https://www.tensorflow.org/tutorials/generative/dcgan
#     '''
#     # print("retrace")
#     noise = tf.random.normal([batch_size, DIM])
#     epsilon = tf.random.uniform(shape=[batch_size, 1, 1], minval=0, maxval=1)
#     ###################################
#     # Train D
#     ###################################
#     with tf.GradientTape(persistent=True) as d_tape:
#         with tf.GradientTape() as gp_tape:
#             fake_image = G([noise], training=True)
#             fake_image_mixed = epsilon * tf.dtypes.cast(real_image, tf.float32) + ((1 - epsilon) * fake_image)
#             fake_mixed_pred = D([fake_image_mixed], training=True)
            
#         # Compute gradient penalty
#         grads = gp_tape.gradient(fake_mixed_pred, fake_image_mixed)
#         grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1,2])) # Originally axis=[1,2]
#         # grad_norms = tf.norm(grads, axis=[1,2])
#         gradient_penalty = tf.reduce_mean(tf.square(grad_norms - 1.))
        
#         fake_pred = D([fake_image], training=True)
#         real_pred = D([real_image], training=True)
        
#         D_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred) + LAMBDA * gradient_penalty
#     # Calculate the gradients for discriminator
#     D_gradients = d_tape.gradient(D_loss,D.trainable_variables)
#     # Apply the gradients to the optimizer
#     D_optimizer.apply_gradients(zip(D_gradients,D.trainable_variables))
#     # Write loss values to tensorboard
#     # tf.print(D_loss)
#     if step % 10 == 0:
#         with file_writer.as_default():
#             tf.summary.scalar('D_loss', tf.reduce_mean(D_loss), step=step)

#     return D_loss, gradient_penalty

# @tf.function
# def WGAN_GP_train_g_step(real_image, batch_size, step):
#     '''
#         One generator training step
        
#         Reference: https://www.tensorflow.org/tutorials/generative/dcgan
#     '''
#     # print("retrace")
#     noise = tf.random.normal([batch_size, DIM])
#     ###################################
#     # Train G
#     ###################################
#     with tf.GradientTape() as g_tape:
#         fake_image = G([noise], training=True)
#         fake_pred = D([fake_image], training=True)
#         G_loss = -tf.reduce_mean(fake_pred)
#     # tf.print(G_loss)
#     # Calculate the gradients for generator
#     G_gradients = g_tape.gradient(G_loss,
#                                             G.trainable_variables)
#     # Apply the gradients to the optimizer
#     G_optimizer.apply_gradients(zip(G_gradients,
#                                                 G.trainable_variables))
#     # Write loss values to tensorboard
#     if step % 10 == 0:
#         with file_writer.as_default():
#             tf.summary.scalar('G_loss', G_loss, step=step)

#     return G_loss, noise

# checkpoint_path = os.path.join("checkpoints", "tensorflow", MODEL_NAME)

# ckpt = tf.train.Checkpoint(generator=G,
#                         discriminator=D,
#                         G_optimizer=G_optimizer,
#                         D_optimizer=D_optimizer)

# ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=40)

# def generate_and_save_images(model, epoch, test_input, figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False):
#     '''
#         Generate images and plot it.
#     '''
#     predictions = model.predict(test_input)
#     utils.save_samples(logdir, predictions, epoch, rev_rna_vocab, annotated=False)
#     # if is_flatten:
#     #     predictions = predictions.reshape(-1, IMG_WIDTH, IMG_HEIGHT, 3).astype('float32')
#     # fig = plt.figure(figsize=figure_size)
#     # for i in range(predictions.shape[0]):
#     #     axs = plt.subplot(subplot[0], subplot[1], i+1)
#     #     plt.imshow(predictions[i] * 0.5 + 0.5)
#     #     plt.axis('off')
#     # if save:
#     #     plt.savefig(os.path.join(OUTPUT_PATH, 'image_at_epoch_{:04d}.png'.format(epoch)))
#     # plt.show()
    

# # if a checkpoint exists, restore the latest checkpoint.
# # if ckpt_manager.latest_checkpoint:
# #     ckpt.restore(ckpt_manager.latest_checkpoint)
# #     latest_epoch = int(ckpt_manager.latest_checkpoint.split('-')[1])
# #     CURRENT_EPOCH = latest_epoch * SAVE_EVERY_N_EPOCH
# #     print ('Latest checkpoint of epoch {} restored!!'.format(CURRENT_EPOCH))

# '''
# load data
# '''

# Train = True
# validate = True

# data = ohe_sequences
# print("Data Shape:")
# print(data.shape)

# if validate:
#     split = len(data) // 12
#     # print(split)
#     valid_data = data[:split]
#     train_data = data[split:]
#     if len(train_data) == 1: train_data = train_data[0]
#     if len(valid_data) == 1: valid_data = valid_data[0]
# else:
#     train_data = data

# train_data = train_data.astype('float32')
# valid_data = valid_data.astype('float32')

# train_seqs = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(BATCH_SIZE)
# valid_seqs = tf.data.Dataset.from_tensor_slices(valid_data).shuffle(2000).batch(split)

# # def feed(data, batch_size=BATCH_SIZE, reuse=True):
# #     num_batches = len(data) // batch_size
# #     if model_type=="mlp":
# #         reshaped_data = np.reshape(data, [data.shape[0], -1])
# #     elif model_type=="resnet":
# #         reshaped_data = data
# #     while True:
# #         for ctr in range(num_batches):
# #             yield reshaped_data[ctr * batch_size : (ctr + 1) * batch_size]
# #         if not reuse and ctr == num_batches - 1:
# #             yield None

# # train_seqs = feed(train_data)
# # valid_seqs = feed(valid_data, reuse=False)

# plot_iter = 100

# current_learning_rate = LR
# trace = True
# n_critic_count = 0

# d_losses = []
# g_losses = []
# gradient_penalties = []
# iterations = 0
# iteration_numbers = []
# iteration_numbers_valid = []
# d_validation_losses = []

# random_name = time.strftime("%Y%m%d-%H%M%S")

# if Train:
#     sample_noise = tf.random.normal([BATCH_SIZE, DIM])
#     generate_and_save_images(G, 0, [sample_noise], figure_size=(12,6), subplot=(3,6), save=False, is_flatten=False)

#     pbar = tqdm(range(EPOCHs))
#     for epoch in pbar:
#         start = time.time()
#         # print('Start of epoch %d' % (epoch,))
#         # Using learning rate decay
#         # current_learning_rate = learning_rate_decay(current_learning_rate)
#         # print('current_learning_rate %f' % (current_learning_rate,))
#         # set_learning_rate(current_learning_rate)
        
#         tdataset = train_seqs.enumerate()

#         for step, tdata in tdataset.as_numpy_iterator():
#             # print(step)
#             current_batch_size = tdata.shape[0]
#             # print(tdata.shape)
#             # Train critic (discriminator)
#             d_loss, gp = WGAN_GP_train_d_step(tdata, batch_size=tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
#             n_critic_count += 1
#             if n_critic_count >= CRITIC_ITERS: 
#                 # Train generator
#                 g_loss, noise = WGAN_GP_train_g_step(tdata, batch_size= tf.constant(current_batch_size, dtype=tf.int64), step=tf.constant(step, dtype=tf.int64))
#                 n_critic_count = 0
            
#             if step % 10 == 0:
#                 print ('.', end='')
        
#         # Clear jupyter notebook cell output
#         clear_output(wait=True)
#         # Using a consistent sample so that the progress of the model is clearly visible.
#         # print("Saving")
        
        
#         if epoch % SAVE_EVERY_N_EPOCH == 0 and epoch != 0:
#             ckpt_save_path = ckpt_manager.save()
#             utils.save_checkpoints(logdir,G,epoch)

#         if epoch % 200 == 0:    
            
#             generate_and_save_images(G, epoch, [sample_noise], figure_size=(12,6), subplot=(3,6), save=True, is_flatten=False)
#             # print ('Saving checkpoint for epoch {} at {}'.format(epoch,
#             #                                                     ckpt_save_path))
        
#         # print ('Time taken for epoch {} is {} sec\n'.format(epoch,
#         #                                                 time.time()-start))
#         # pbar.refresh()
#         os.system('clear')
#         pbar.set_description(f'GP:{gp:.4f}, D_loss:{d_loss:.4f}, G_loss:{g_loss:.4f}')
#         if iterations % 5 == 0:
#             iteration_numbers.append(iterations)
#             g_losses.append(-g_loss)
#             d_losses.append(-d_loss)
#             gradient_penalties.append(gp)
#             plot(iteration_numbers, d_losses, logdir, 'discriminator_loss', xlabel="Iteration", ylabel="Discriminator Cost")
#             plot(iteration_numbers, g_losses, logdir, 'generator_loss', xlabel="Iteration", ylabel="Generator Cost")
#             plot(iteration_numbers, gradient_penalties, logdir, 'gradient_penalty', xlabel="Iteration", ylabel="Gradient Penalty")

#         iterations+=1
#         if iterations % 20 == 0:
#             iteration_numbers_valid.append(iterations)
#             fake_image_valid = G([noise], training=True)
#             fake_pred_valid = D([fake_image_valid], training=True)
#             real_pred_valid = D([tf.convert_to_tensor(list(valid_seqs.as_numpy_iterator())[0])], training=True)
            
#             D_loss_valid = tf.reduce_mean(fake_pred_valid) - tf.reduce_mean(real_pred_valid)
#             d_validation_losses.append(-D_loss_valid)

#             plot_valid(iteration_numbers_valid, d_validation_losses, iteration_numbers, d_losses, logdir, 'validation_loss', xlabel="Iteration", ylabel="D Validation Loss")
#             # valid_seqs.

#     # Save at final epoch
#     ckpt_save_path = ckpt_manager.save()
#     print ('Saving checkpoint for epoch {} at {}'.format(EPOCHs,
#                                                             ckpt_save_path))


