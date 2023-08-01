import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, ZeroPadding1D,\
     Flatten, BatchNormalization, AveragePooling1D, Dense, Activation, Add, Softmax, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from tensorflow.keras.initializers import RandomUniform

import tensorflow.keras.backend as K

def resblock2(inputs,num_channels):
  res = inputs
    #first block 
  c = num_channels
  # print(type(res))
  x = Activation(activations.relu)(res)
  x = Conv1D(c, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
  # x = BatchNormalization()(x)

  x = Activation(activations.relu)(x)
  x = Conv1D(c, kernel_size=5, strides=1, padding='same', kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001))(x)
  # x = BatchNormalization()(x)
  
  # add the input 
  x = Add()([x * 0.3, inputs])
  return x

def resnet_g2(dim, num_channels, seq_len, vocab_size, annotated=False, res_layers=2, batch_size=64):
  output_size = seq_len * num_channels
  input_data = Input(shape=(dim))

  x = input_data
  x = Dense(output_size,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.001))(x)

  x = Reshape(target_shape=(-1,num_channels))(x)
  
  for layer in range(res_layers):
    x = resblock2(x,num_channels)

  x = Conv1D(vocab_size,1,padding='same', kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01))(x)

  x = Softmax()(x)
  # print(x.shape)

  model = Model(inputs=input_data, outputs=x, name='Generator')

  return model

def resnet_d2(num_channels, seq_len, vocab_size, batch_size=64, res_layers=2):
  input_size = seq_len * vocab_size
  input_data = Input(shape=(seq_len,vocab_size))

  x = input_data
  x = Conv1D(num_channels,kernel_size=1,padding='same')(x)
  
  for layer in range(res_layers):
    x = resblock2(x,num_channels)

  x = tf.keras.layers.Flatten()(x)

  x = Dense(1)(x)

  model = Model(inputs=input_data, outputs=x, name='Discriminator')

  return model

def wasserstein_loss( y_true, y_pred):
  return K.mean(y_true * y_pred)

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
  """
  Computes gradient penalty based on prediction and weighted real / fake samples
  """
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