# -*- coding: utf-8 -*-
"""DP-GAN.py"""

seed=1
import os
os.environ['PYTHONHASHSEED'] = str(seed)
# For working on GPUs from "TensorFlow Determinism"
os.environ["TF_DETERMINISTIC_OPS"] = str(seed)
import numpy as np
np.random.seed(seed)
import random
random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)

from __future__ import print_function, division
from keras.datasets import mnist
from tensorflow.keras.optimizers import RMSprop
from functools import partial
import tensorflow as tf
import argparse
import random
import seaborn as sns
import os
import numpy as np
import keras
from tensorflow.keras import backend as K
from functools import partial
from google.colab import drive
from google.colab import files
from __future__ import print_function, division
import os
from sklearn.linear_model import LinearRegression
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, LSTM
from keras.layers import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pandas as pd
import io
from keras.models import load_model
import time
from scipy.stats import pearsonr
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras import losses
import keras.backend as K

drive.mount('//drive')

random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

data = pd.read_csv('data.csv', sep = ',', na_values=['(NA)']).fillna(0)
data = np.array(data)
data = data.reshape(1262423,9)

# correlation real
corr_data = pd.DataFrame(data)
corr_data = corr_data.corr()
corr_data = np.array(corr_data)
corr_data = corr_data.reshape(len(corr_data)*len(corr_data))
# model
real_fit = LinearRegression().fit(data[:,1:9], data[:,0])

from sklearn.preprocessing import MinMaxScaler
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
scaler0 = MinMaxScaler(feature_range= (-1, 1))
scaler0 = scaler0.fit(data)
data = scaler0.transform(data)
print(data)
data = data.reshape(1262423,9)

class GAN():
    def __init__(self, privacy):
      self.img_rows = 1
      self.img_cols = 9
      self.img_shape = (self.img_cols,)
      self.latent_dim = (9)

      optimizer = keras.optimizers.Adam()
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
      if privacy == True:
        print("using differential privacy")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=lr),
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE), metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer= optimizer)


    def build_generator(self):
      model = Sequential()
      model.add(Dense(self.latent_dim, input_dim=self.latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(1024, input_shape=self.img_shape))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(self.latent_dim))
      model.add(Activation("tanh"))

      model.summary()

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(1024, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, iterations, batch_size, sample_interval, early_stopping, early_stopping_value, generator_losses = [], discriminator_acc = [], correlations = [], accuracy = [], MAPD_collect = [],MSE_collect = [], MAE_collect = []):
      # Adversarial ground truths

      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      corr = 0
      MAPD = 0
      MSE = 0
      MAE = 0
      #fake += 0.05 * np.random.random(fake.shape)
      #valid += 0.05 * np.random.random(valid.shape)

      for epoch in range(iterations):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose = False)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # collect losses
            discriminator_acc = np.append(discriminator_acc, 100*d_loss[1])
            generator_losses = np.append(generator_losses, g_loss)


            if epoch % sample_interval == 0:
              # Plot the progress and save model
              self.generator.save("model.h5")
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, corr: %f, MAPD: %f, MSE: %f, MAE: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, corr, MAPD, MSE, MAE))


              # ---------------------
              #  Summary stats
              # ---------------------

              # Gets correlation, MSE, MAPD, MAE, and others.
              sample_size = 20000
              idx = np.random.randint(0, data.shape[0], sample_size)
              noise = np.random.normal(0, 1, (sample_size, self.latent_dim))
              gen_imgs1 = self.generator.predict(noise, verbose = False)
              gen_imgs1 = scaler0.inverse_transform(gen_imgs1)

              # model
              GAN_fit = LinearRegression().fit(gen_imgs1[:,1:24], gen_imgs1[:,0])
              MAPD = np.mean(abs((real_fit.coef_-GAN_fit.coef_)/real_fit.coef_))*100
              MSE = np.mean((real_fit.coef_-GAN_fit.coef_)**2)
              MAE = np.mean(abs(real_fit.coef_-GAN_fit.coef_))


              # correlation
              corr_artificial1 = gen_imgs1.reshape(sample_size,self.latent_dim)
              corr_artificial = pd.DataFrame(corr_artificial1)
              corr_artificial = corr_artificial.corr()
              corr_artificial = np.array(corr_artificial)
              corr_artificial = corr_artificial.reshape(len(corr_artificial)*len(corr_artificial))
              try:
                corr = pearsonr(corr_artificial,corr_data)[0]
              except:
                print("pearson r error")
              # collect
              correlations = np.append(correlations,corr)
              MAPD_collect = np.append(MAPD_collect, MAPD)
              MSE_collect = np.append(MSE_collect, MSE)
              MAE_collect = np.append(MAE_collect, MAE)
              pd.DataFrame(correlations).to_csv("cor.csv", index = False)
              pd.DataFrame(MAPD_collect).to_csv("MAPD.csv", index = False)
              pd.DataFrame(MSE_collect).to_csv("MSE.csv", index = False)
              pd.DataFrame(MAE_collect).to_csv("MAE.csv", index = False)

if __name__ == '__main__':
  random.seed(1)
  np.random.seed(1)
  tf.random.set_seed(1)
  gan = GAN(privacy = False)
  gan.train(iterations=100000, batch_size=128, sample_interval=1000, early_stopping = False, early_stopping_value = 50)

import numpy as np
from keras.models import load_model
generator = load_model('model.h5')
generated_data = []
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
noise = np.random.normal(0, 1, (1262423, 9))
# Generate a batch of new images
gen_imgs = generator.predict(noise)
gen_imgs = gen_imgs.reshape(1262423, 9)
gen_imgs = pd.DataFrame(gen_imgs)
gen_imgs = scaler0.inverse_transform(gen_imgs)
gen_imgs = pd.DataFrame(gen_imgs)
gen_imgs.to_csv("model.csv", index = False)
print("stored")

"""# with differential privacy

"""

!pip install tensorflow_privacy --quiet

"""
**settings for differential privacy**

*noise multipliers voor n = 1,262:*
- 4149 = 0.01, 1285 = 0.05, 173.2 = 0.5, 95 = 1, 37.4 = 3, 11.67 =13.

*noise multipliers voor n = 12,624:*
- 679 = 0.01, 174 = 0.05, 21.4 = 0.5, 11.4 = 1, 4.38 = 3, 1.45 = 13.

*noise multipliers voor n = 1,262,423*
- noise multiplier is 1993 = 0.01, 3.73179 = 0.05, 1.1047 = 0.5, 2.639 = 0.1, 0.8165 = 1, 0.554 = 3, 0.3685 = 13.
"""

from absl import app
from absl import flags
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer, DPKerasAdamOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy

# hyperparams
N = 1262423

# epoch is the number of iterations it takes to completely cover a data set.
# how many iterations would it take to cover a complete data set: dataset/batch_size = (3333/33) = 101
# one epoch is the number of iterations it takes to cover a data set. one epoch = 101 iterations, thus epochs = iterations/101
batch_size = 128
iterations = 100000
num_microbatches = 128

epochs = iterations/(N/batch_size)
print("the number of epochs is " + str(epochs))

lr = 0.001
## noise_multiplier has to be at least .3 to obtain some privacy guarantees.
## the higher the noise_multiplier, the better the privacy, but lower accuracy.
noise_multiplier = 1993

## if l2_norm_clip < ||gradient||2, gradient is preserved. if l2_norm_clip > ||gradient||2, then gradient is divided by l2_norm_clip
l2_norm_clip = 4 # see abadi et al 2016

delta= 1/N

compute_dp_sgd_privacy(N, batch_size, noise_multiplier,
                         epochs, delta)

class GAN():
    def __init__(self, privacy):
      self.img_rows = 1
      self.img_cols = 9
      self.img_shape = (self.img_cols,)
      self.latent_dim = (9)

      optimizer = Adam()
      self.discriminator = self.build_discriminator()
      self.discriminator.compile(loss='binary_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])
      if privacy == True:
        print("using differential privacy")
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer=DPKerasAdamOptimizer(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=noise_multiplier,
            num_microbatches=num_microbatches,
            learning_rate=lr),
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE), metrics=['accuracy'])

      # Build the generator
      self.generator = self.build_generator()

      # The generator takes noise as input and generates imgs
      z = Input(shape=(self.latent_dim,))
      img = self.generator(z)

      # For the combined model we will only train the generator
      self.discriminator.trainable = False

      # The discriminator takes generated images as input and determines validity
      valid = self.discriminator(img)

      # The combined model  (stacked generator and discriminator)
      # Trains the generator to fool the discriminator
      self.combined = Model(z, valid)
      self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


    def build_generator(self):
      model = Sequential()
      model.add(Dense(self.latent_dim, input_dim=self.latent_dim))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(64, input_shape=self.img_shape))
      model.add(LeakyReLU(alpha=0.2))
      #model.add(BatchNormalization())
      model.add(Dense(self.latent_dim))
      model.add(Activation("tanh"))

      model.summary()

      noise = Input(shape=(self.latent_dim,))
      img = model(noise)
      return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(64, input_shape=self.img_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, iterations, batch_size, sample_interval, early_stopping, early_stopping_value, generator_losses = [], discriminator_acc = [], correlations = [], accuracy = [], MAPD_collect = [],MSE_collect = [], MAE_collect = []):
      # Adversarial ground truths
      valid = np.ones((batch_size, 1))
      fake = np.zeros((batch_size, 1))
      corr = 0
      MAPD = 0
      MSE = 0
      MAE = 0
      #fake += 0.05 * np.random.random(fake.shape)
      #valid += 0.05 * np.random.random(valid.shape)

      for epoch in range(iterations):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, data.shape[0], batch_size)
            imgs = data[idx]

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images
            gen_imgs = self.generator.predict(noise, verbose = False)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator (to have the discriminator label samples as valid)

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)

            # collect losses
            discriminator_acc = np.append(discriminator_acc, 100*d_loss[1])
            generator_losses = np.append(generator_losses, g_loss)


            if epoch % sample_interval == 0:
              # Plot the progress and save model
              self.generator.save("model.h5")
              print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, corr: %f, MAPD: %f, MSE: %f, MAE: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss, corr, MAPD, MSE, MAE))



if __name__ == '__main__':
  random.seed(1)
  np.random.seed(1)
  tf.random.set_seed(1)
  gan = GAN(privacy = True)
  gan.train(iterations=100001, batch_size=128, sample_interval=1, early_stopping = False, early_stopping_value = 50)

import numpy as np
from keras.models import load_model
generator = load_model('data_without_privacy_e_3.h5')
generated_data = []
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
noise = np.random.normal(0, 1, (1262423, 9))
# Generate a batch of new images
gen_imgs = generator.predict(noise)
gen_imgs = gen_imgs.reshape(1262423, 9)
gen_imgs = pd.DataFrame(gen_imgs)
gen_imgs = scaler0.inverse_transform(gen_imgs)
gen_imgs = pd.DataFrame(gen_imgs)
gen_imgs.to_csv("data_without_privacy_e_3.csv", index = False)
print("stored")
