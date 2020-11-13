import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import LeakyReLU, Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image as PILImage
import glob
import cv2
import random

from .utils import load_frames

class GAN():
    def __init__(self, config):
        self.config = config
    
    def plot_examples(self, imgs):
        count = 10

        plt.figure(figsize=(15,3))
        for i in range(count):
            plt.subplot(2, count // 2, i+1)
            plt.imshow(imgs[i])
            plt.axis('off')
            
        plt.tight_layout()
        plt.show()
    
    def _build_discriminator(self):
        model = Sequential()
        input_shape = (64, 64, 3)
        dropout_prob = 0.4

        model.add(Conv2D(64, 5, strides=2, input_shape=input_shape, padding='same'))
        model.add(LeakyReLU())
        
        model.add(Conv2D(128, 5, strides=2, padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_prob))
        
        model.add(Conv2D(256, 5, strides=2, padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_prob))
        
        model.add(Conv2D(512, 5, strides=2, padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(dropout_prob))
        
        model.add(Flatten())
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        
        return model

    def _build_generator(self):
        
        model = Sequential()
        dropout_prob = 0.4
        
        model.add(Dense(8*8*256, input_dim=100))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((8,8,256)))
        model.add(Dropout(dropout_prob))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D())
        model.add(Conv2D(128, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        
        model.add(UpSampling2D())
        model.add(Conv2D(64, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        
        model.add(Conv2D(32, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        
        model.add(Conv2D(3, 5, padding='same'))
        model.add(Activation('sigmoid'))
            
        return model
    
    def fit(self):
        net_discriminator = self._build_discriminator()
        net_generator = self._build_generator()

        optim_discriminator = RMSprop(lr=0.0002, clipvalue=1.0, decay=6e-8)
        model_discriminator = Sequential()
        model_discriminator.add(net_discriminator)
        model_discriminator.compile(loss='binary_crossentropy', optimizer=optim_discriminator, metrics=['accuracy'])

        optim_adversarial = Adam(lr=0.0001, clipvalue=1.0, decay=3e-8)
        model_adversarial = Sequential()
        model_adversarial.add(net_generator)

        # Disable layers in discriminator
        for layer in net_discriminator.layers:
            layer.trainable = False

        model_adversarial.add(net_discriminator)
        model_adversarial.compile(loss='binary_crossentropy', optimizer=optim_adversarial, metrics=['accuracy'])
    
        batch_size = 128

        vis_noise = np.random.uniform(-1.0, 1.0, size=[16, 100])

        loss_adv = []
        loss_dis = []
        acc_adv = []
        acc_dis = []
        plot_iteration = []
 
        for i in range(0, self.config.epochs):    
            images_train = load_frames(n_samples=batch_size, source_folder=self.config.source_folder)
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = net_generator.predict(noise)
            
            x = np.concatenate((images_train, images_fake))
            y = np.ones([2 * batch_size, 1])
            y[batch_size:, :] = 0

            # Train discriminator for one batch
            d_stats = model_discriminator.train_on_batch(x, y)
            
            y = np.ones([batch_size, 1])

            # Train the generator for a number of times
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            a_stats = model_adversarial.train_on_batch(noise, y)

            if i % 25 == 0:
                print(f'iii = {i}')
                images = net_generator.predict(vis_noise)
                
                # Map back to original range
                images = (images + 1 ) * 0.5
                
                plt.figure(figsize=(10,10))
                for im in range(images.shape[0]):
                    plt.subplot(4, 4, im+1)
                    image = images[im, :, :, :]
                    image = np.reshape(image, [64, 64,3])
                    plt.imshow(image)
                    plt.axis('off')
                    
                plt.tight_layout()
                plt.savefig(r'crash-generated/{}.png'.format(i))
