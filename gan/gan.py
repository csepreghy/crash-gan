import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

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
    
    def discriminator():
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