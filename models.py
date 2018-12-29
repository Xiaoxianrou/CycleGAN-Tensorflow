import tensorflow as tf
import numpy as np
import random

from PIL import Image
from layers import *

REAL_LABEL = 1
FAKE_LABEL = 0

INITIAL_ITER = 100
DECAY_ITER = 100
INITIAL_LR = 0.0002


class Generator(object):
    def __init__(self, name, is_train, norm='instance', activation='relu', image_size=128):
        print('Init Generator', name)
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
        print('Init Discriminator', name)
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


class HistoryQueue(object):
    def __init__(self, shape=[128, 128, 3], size=50):
        self._size = size
        self._shape = shape
        self._count = 0
        self._queue = []

    def query(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        if self._size == 0:
            return image
        if self._count < self._size:
            self._count += 1
            self._queue.append(image)
            return image

        p = random.random()
        if p > 0.5:
            idx = random.randrange(0, self._size)
            ret = self._queue[idx]
            self._queue[idx] = image
            return ret
        else:
            return image


class CycleGAN(object):
    def __init__(self, args):
        self._epoch_num = args.epoch_num
        self._batch_size = args.batch_size
        self._image_size = args.image_size
        self._cycle_loss_lambda = args.cycle_loss_lambda

        self.image_shape = [self._image_size, self._image_size, 3]
        self.is_train = tf.placeholder(tf.bool, name='is_train')
        self.lr = tf.placeholder(tf.float32, name='learning_rate')

        self.image_a = tf.placeholder(tf.float32, [self._batch_size] + self.image_shape, name='image_a')
        self.image_b = tf.placeholder(tf.float32, [self._batch_size] + self.image_shape, name='image_b')
        self.history_fake_a = tf.placeholder(tf.float32, [None] + self.image_shape, name='history_fake_a')
        self.history_fake_b = tf.placeholder(tf.float32, [None] + self.image_shape, name='history_fake_b')

        self.build()
        self.summary_op = tf.summary.merge_all()

    def build(self):
        # build Generator
        G_ab = Generator('G_ab', is_train=self.is_train, norm='instance', activation='relu',
                         image_size=self._image_size)
        G_ba = Generator('G_ba', is_train=self.is_train, norm='instance', activation='relu',
                         image_size=self._image_size)

        # build Discriminator
        D_a = Discriminator('D_a', is_train=self.is_train, norm='instance', activation='leaky')
        D_b = Discriminator('D_b', is_train=self.is_train, norm='instance', activation='leaky')

        # Generate images (a->b->a and b->a->b)
        self.image_ab = G_ab(self.image_a)
        self.image_aba = G_ba(self.image_ab)
        self.image_ba = G_ba(self.image_b)
        self.image_bab = G_ab(self.image_ba)

        # Discriminate real/fake images
        D_real_a = D_a(self.image_a)
        D_fake_a = D_a(self.image_ba)
        D_real_b = D_b(self.image_b)
        D_fake_b = D_b(self.image_ab)
        D_history_fake_a = D_a(self.history_fake_a)
        D_history_fake_b = D_b(self.history_fake_b)

        # Least squre loss for GAN discriminator
        self.loss_D_a = tf.contrib.gan.losses.wargs.least_squares_discriminator_loss(D_real_a, D_history_fake_a,
                                                                                     real_label=REAL_LABEL,
                                                                                     fake_label=FAKE_LABEL)
        self.loss_D_b = tf.contrib.gan.losses.wargs.least_squares_discriminator_loss(D_real_b, D_history_fake_b,
                                                                                     real_label=REAL_LABEL,
                                                                                     fake_label=FAKE_LABEL)
        # Least squre loss for GAN generator
        loss_G_ab_ls = tf.contrib.gan.losses.wargs.least_squares_generator_loss(D_fake_b, real_label=REAL_LABEL)
        loss_G_ba_ls = tf.contrib.gan.losses.wargs.least_squares_generator_loss(D_fake_a, real_label=REAL_LABEL)

        # L1 norm for cycle loss
        loss_cycle_aba = tf.reduce_mean(tf.abs(self.image_a - self.image_aba))
        loss_cycle_bab = tf.reduce_mean(tf.abs(self.image_b - self.image_bab))
        self.loss_cycle = self._cycle_loss_lambda * (loss_cycle_aba + loss_cycle_bab)

        # GAN generator loss
        self.loss_G_ab = loss_G_ab_ls + self.loss_cycle
        self.loss_G_ba = loss_G_ba_ls + self.loss_cycle

        # Optimizer
        self.optimizer_D_a = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(self.loss_D_a, var_list=D_a.var_list)
        self.optimizer_D_b = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(self.loss_D_b, var_list=D_b.var_list)
        self.optimizer_G_ab = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(self.loss_G_ab, var_list=G_ab.var_list)
        self.optimizer_G_ba = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5) \
            .minimize(self.loss_G_ba, var_list=G_ba.var_list)

    def train(self, sess, summary_writer, data_A, data_B):
        print('Start training')
        data_size = min(len(data_A), len(data_B))
        batch_num = data_size // self._batch_size
        lr_decay = INITIAL_LR / DECAY_ITER

        history_a = HistoryQueue(shape=self.image_shape, size=50)
        history_b = HistoryQueue(shape=self.image_shape, size=50)

        init_all = tf.initializers.global_variables()
        sess.run(init_all)

        # Training epoch loop
        for epoch in range(self._epoch_num):
            print("In the epoch ", epoch)
            # shuffle datasets
            random.shuffle(data_A)
            random.shuffle(data_B)
            batch_idxs = min(len(data_A), len(data_B)) // self._batch_size

            # Training batch loop
            for idx in range(0, batch_idxs):
                # initial input
                image_a = np.stack(data_A[idx * self._batch_size:(idx + 1) * self._batch_size])
                image_b = np.stack(data_B[idx * self._batch_size:(idx + 1) * self._batch_size])
                fake_a, fake_b = sess.run([self.image_ba, self.image_ab], feed_dict={self.image_a: image_a,
                                                                                     self.image_b: image_b,
                                                                                     self.is_train: True})
                fake_a = history_a.query(fake_a)
                fake_b = history_b.query(fake_b)

                # fetch result
                if epoch > INITIAL_ITER:
                    learning_rate = max(0.0, INITIAL_LR - (epoch - INITIAL_ITER) * lr_decay)
                else:
                    learning_rate = INITIAL_LR

                fetches = [self.loss_D_a, self.loss_D_b, self.loss_G_ab, self.loss_G_ba, self.loss_cycle,
                           self.optimizer_D_a, self.optimizer_D_b, self.optimizer_G_ab, self.optimizer_G_ba]
                fetched = sess.run(fetches, feed_dict={self.image_a: image_a,
                                                       self.image_b: image_b,
                                                       self.is_train: True,
                                                       self.lr: learning_rate,
                                                       self.history_fake_a: fake_a,
                                                       self.history_fake_b: fake_b})
            if epoch // 50 != 0:
                self.test(sess, data_A, data_B)

    def test(self, sess, data_A, data_B):
        for idx in range(3):
            fetches = [self.image_ba, self.image_bab]
            image_b = np.expand_dims(data_B[idx], axis=0)
            image_ba, image_bab = sess.run(fetches, feed_dict={self.image_b: image_b, self.is_train: False})
            # images = np.concatenate((image_b, image_ba, image_bab), axis=2)
            # images = np.squeeze(images, axis=0)
            image_ba = np.squeeze(image_ba, axis=0)
            img_ba = Image.fromarray((image_ba*255.0).astype('uint8'))
            img_ba.save()
            # print(images)
