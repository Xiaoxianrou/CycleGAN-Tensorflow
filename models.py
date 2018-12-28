import tensorflow as tf
import numpy as np
import os

from .layers import *


class Generator(object):
    def __init__(self, name, is_train, norm='instance', activation='relu', image_size=128):
        print('Init Generator %s', name)
        self.name = name
        self._train = is_train
        self._norm = norm
        self._act = activation
        self._reuse = False
        self.res_block_num = 6 if image_size == 128 else 9

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            G = conv_block(input, 32, 'c7s1-32', 7, 1, self._train, self._reuse, self._norm, self._act, pad='REFLECT')
            G = conv_block(G, 64, 'd64', 3, 2, self._train, self._reuse, self._norm, self._act)
            G = conv_block(G, 128, 'd128', 3, 2, self._train, self._reuse, self._norm, self._act)
            for i in range(self.res_block_num):
                G = res_block(G, 128, 'R128_{}'.format(i), self._train, self._reuse, self._norm)
            G = deconv_block(G, 64, 'u64', 3, 2, self._train, self._reuse, self._norm, self._act)
            G = deconv_block(G, 32, 'u32', 3, 2, self._train, self._reuse, self._norm, self._act)
            G = conv_block(G, 3, 'c7s1-3', 7, 1, self._train, self._reuse, norm=None, activation='tanh', pad='REFLECT')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return G


class Discriminator(object):
    def __init__(self, name, is_train, norm='instance', activation='leaky'):
        print('Init Discriminator %s', name)
        self.name = name
        self._is_train = is_train
        self._norm = norm
        self._act = activation
        self._reuse = False

    def __call__(self, input):
        with tf.variable_scope(self.name, reuse=self._reuse):
            D = conv_block(input, 64, 'C64', 4, 2, self._is_train, self._reuse, norm=None, activation=self._act)
            D = conv_block(D, 128, 'C128', 4, 2, self._is_train, self._reuse, self._norm, self._act)
            D = conv_block(D, 256, 'C256', 4, 2, self._is_train, self._reuse, self._norm, self._act)
            D = conv_block(D, 512, 'C512', 4, 2, self._is_train, self._reuse, self._norm, self._act)
            D = conv_block(D, 1, 'C1', 4, 1, self._is_train, self._reuse, norm=None, activation=None, bias=True)
            D = tf.reduce_mean(D, axis=[1, 2, 3])

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return D


class CycleGAN():
    def __init__(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
