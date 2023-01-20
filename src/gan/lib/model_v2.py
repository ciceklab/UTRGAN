
# example of a wgan for generating handwritten digits
import tensorflow as tf
from numpy import expand_dims
from numpy import mean
from numpy import ones
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras import backend
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv1DTranspose
from tensorflow.keras.layers import LeakyReLU, Softmax	
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, RandomUniform, Zeros
from tensorflow.keras.constraints import Constraint
from matplotlib import pyplot

from tensorflow.keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import datetime


# tensorboard debug
tf.debugging.experimental.enable_dump_debug_info('~/ml/gan/dev/shot0/lib/tlog', tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='~/ml/gan/dev/shot0/lib/tlog', histogram_freq=1)


# clip model weights to a given hypercube
class ClipConstraint(Constraint):
	# set clip value when initialized
	def __init__(self, clip_value):
		self.clip_value = clip_value
 
	# clip model weights to hypercube
	def __call__(self, weights):
		return backend.clip(weights, -self.clip_value, self.clip_value)
 
	# get the config
	def get_config(self):
		return {'clip_value': self.clip_value}
 
# calculate wasserstein loss
def wasserstein_loss(y_true, y_pred):
	return backend.mean(y_true * y_pred)

# leaky relu
def leaky_relu(input_, alpha=0.3):
	return tf.math.maximum(alpha * input_, input_)

# define the standalone critic model
def define_critic(dim = 25, seq_len = 256 , enc_len =5, num_layers=2):
	kernel_init = RandomNormal(stddev=0.02)
	# bias initialization
	# bias_init = RandomUniform(minval = .0, maxval = .0)
	bias_init = Zeros()
	# weight constraint
	# define model
	model = Sequential()

	dim = 25

	seq_len = 256

	vocab_size = 5

	input_size = seq_len * vocab_size

	model.add(Conv1D(dim, (4), input_shape=(256,5), strides=1, padding='same', use_bias=True, kernel_initializer=kernel_init,bias_initializer=bias_init))
	# model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	''' Add Residual Connections Here to the Loop '''

	num_layers = 2

	num_layers_stride = 5


	# Add Layers
	for layer in range(num_layers):

		model.add(Conv1D(dim, (4), strides=1, padding='same', kernel_initializer=kernel_init))

	for layer in range(num_layers_stride):

		model.add(Conv1D(dim, (4), strides=2, padding='same', kernel_initializer=kernel_init))

	# model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	# scoring, linear activation
	model.add(Flatten())
	model.add(Dense(1))
	# compile model
	opt = RMSprop(learning_rate=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
 
# define the standalone generator model
def define_generator(dim=25, input_size=32 * 8, enc_len = 5, num_layers=3):

	input_size = 32 * 8

	vocab_size = 5

	n_nodes = input_size * vocab_size

	kernel_init = RandomNormal(stddev=0.02)

	dim = 25

	vocab_size = enc_len
	
	# define model
	model = Sequential()

	model.add(Flatten())

	n_nodes = input_size 

	model.add(Dense(n_nodes, kernel_initializer=kernel_init, bias_initializer=Zeros(), input_dim=1))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((32,8)))

	model.add(Conv1D(dim, (4), strides=1, padding='same', kernel_initializer=kernel_init))
	model.add(LeakyReLU(alpha=0.2))


	for layer in range(num_layers):
		model.add(Conv1DTranspose(dim, (4), strides=2, padding='same', kernel_initializer=RandomNormal(stddev=0.02),bias_initializer=Zeros()))
		# model.add(BatchNormalization())
		model.add(LeakyReLU(alpha=0.2))

	model.add(Conv1DTranspose(dim, (4), strides=1, padding='same', kernel_initializer=RandomNormal(stddev=0.02),bias_initializer=Zeros()))
	# model.add(BatchNormalization())
	model.add(LeakyReLU(alpha=0.2))

	model.add(Reshape((-1,256,5))) 

	model.add(Softmax())
	return model
	
# define the combined generator and critic model, for updating the generator
def define_gan(generator, critic):
	# make weights in the critic not trainable
	for layer in critic.layers:
		if not isinstance(layer, BatchNormalization):
			layer.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(generator)
	# add the critic
	model.add(critic)
	# compile model
	opt = RMSprop(lr=0.00005)
	model.compile(loss=wasserstein_loss, optimizer=opt)
	return model
 

def load_utr_data():
	data_path = "/home/sina/ml/gan/dev/shot0/selected.csv"
	data = pd.read_csv(data_path)
	UTRdf = data["utrs"]
	UTRlen = data["length"].to_list()
	max_length = max(UTRlen)
	UTRdf = UTRdf.values

	sequences = list(UTRdf)
	sequences = [x.upper() for x in sequences]

	rna_vocab = {"A":0,
				"C":1,
				"G":2,
				"U":3,
				"*":4}

	rev_rna_vocab = {v:k for k,v in rna_vocab.items()}

	def one_hot_encode(seq, SEQ_LEN=256):
		mapping = dict(zip("ACGT*", range(5)))    
		seq2 = [mapping[i] for i in seq]
		if len(seq2) < SEQ_LEN:
			extra = [np.eye(5)[4]] * (SEQ_LEN - len(seq2))
			return np.vstack([np.eye(5)[seq2] , extra])
		return np.eye(5)[seq2]

	ohe_sequences = np.asarray([one_hot_encode(x) for x in sequences])
	return ohe_sequences

def load_real_samples():
	# load dataset
	X = load_utr_data()

	return X
 
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]

	return X
 
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
 
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	return X
 
# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, latent_dim, n_samples=100):
	# prepare fake examples
	
	X = generate_fake_samples(g_model, latent_dim, n_samples)
	
	filename1 = 'generated_seqs_%04d.txt' % (step+1)
	with open(filename1, 'w') as f:
		for sample in X:
			f.write(sample)
			f.write('\n')

	# save the samples to file

	# save the generator model
	# filename2 = 'model_%04d.h5' % (step+1)
	# g_model.save(filename2)
	# print('>Saved: %s and %s' % (filename1, filename2))
 
# create a line plot of loss for the gan and save to file
def plot_history(d1_hist, d2_hist, g_hist):
	# plot history
	pyplot.plot(d1_hist, label='crit_real')
	pyplot.plot(d2_hist, label='crit_fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	pyplot.savefig('plot_line_plot_loss.png')
	pyplot.close()
 
# train the generator and critic
def train(g_model, c_model, gan_model, dataset, latent_dim, n_epochs=250, n_batch=64, n_critic=5):
		
	# calculate the number of batches per training epoch
	bat_per_epo = int(dataset.shape[0] / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# calculate the size of half a batch of samples
	half_batch = int(n_batch / 2)
	# lists for keeping track of loss
	c1_hist, c2_hist, g_hist = list(), list(), list()
	# manually enumerate epochs
	for i in range(n_steps):
		# update the critic more than the generator
		c1_tmp, c2_tmp = list(), list()
		for _ in range(n_critic):
			# get randomly selected 'real' samples
			X_real = generate_real_samples(dataset, half_batch)
			y_real = ones((half_batch, 1))
		
			
			# update critic model weights
			c_loss1 = c_model.train_on_batch(X_real, y_real)
			c1_tmp.append(c_loss1)
			# generate 'fake' examples
			X_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			y_fake = -ones((half_batch, 1))
			# update critic model weights
			c_loss2 = c_model.train_on_batch(X_fake, y_fake)
			c2_tmp.append(c_loss2)
		# store critic loss
		c1_hist.append(mean(c1_tmp))
		c2_hist.append(mean(c2_tmp))
		# prepare points in latent space as input for the generator
		X_gan = generate_latent_points(latent_dim, n_batch)
		# create inverted labels for the fake samples
		y_gan = -ones((n_batch, 1))
		# update the generator via the critic's error
		g_loss = gan_model.train_on_batch(X_gan, y_gan)
		g_hist.append(g_loss)
		# summarize loss on this batch
		print('>%d, c1=%.3f, c2=%.3f g=%.3f' % (i+1, c1_hist[-1], c2_hist[-1], g_loss))
		# evaluate the model performance every 'epoch'
		if (i+1) % bat_per_epo == 0:
			summarize_performance(i, g_model, latent_dim)
	# line plots of loss
	plot_history(c1_hist, c2_hist, g_hist)
 

'''
Set seed
'''
seed = 35
np.random.seed(seed)
tf.random.set_seed(seed)


# latent_vars = tf.Variable(tf.random.normal(shape=[BATCH_SIZE, DIM], seed=seed), name='latent_vars')

# size of the latent space
latent_dim =  32

# layer size
gan_dim = 256

# create the critic
critic = define_critic()
# create the generator
generator = define_generator(latent_dim)
# create the gan
gan_model = define_gan(generator, critic)
# load image data
dataset = load_real_samples()
print(dataset.shape)
# train model
train(generator, critic, gan_model, dataset, latent_dim)


#########################################################################
