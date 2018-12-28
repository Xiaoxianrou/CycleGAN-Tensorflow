import tensorflow as tf


def norm_layer(input, is_train, reuse=True, norm=None):
    assert norm in ['instance', 'batch', None]
    if norm == 'instance':
        with tf.variable_scope('instance_norm', reuse=reuse):
            eps = 1e-5
            out = tf.contrib.layers.instance_norm(input, epsilon=eps)
    elif norm == 'batch':
        with tf.variable_scope('batch_norm', reuse=reuse):
            out = tf.layers.batch_normalization(input, decay=0.99, training=is_train)
    else:
        out = input

    return out


def act_layer(input, activation=None):
    assert activation in ['relu', 'leaky', 'tanh', 'sigmoid', None]
    if activation == 'relu':
        return tf.nn.relu(input)
    elif activation == 'leaky':
        return tf.nn.leaky_relu(input, alpha=0.2)
    elif activation == 'tanh':
        return tf.tanh(input)
    elif activation == 'sigmoid':
        return tf.sigmoid(input)
    else:
        return input


def conv2d(input, filters_num, filter_size, stride, reuse=False,
           pad='SAME', dtype=tf.float32, bias=False):
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, input.get_shape()[3], filters_num]

    w = tf.get_variable('w', filter_shape, dtype, tf.initializers.random_normal(0.0, 0.02))
    if pad == 'REFLECT':
        p = (filter_size - 1) // 2
        x = tf.pad(input, [[0, 0], [p, p], [p, p], [0, 0]], 'REFLECT')
        conv = tf.nn.conv2d(x, w, stride_shape, padding='VALID')
    else:
        assert pad in ['SAME', 'VALID']
        conv = tf.nn.conv2d(input, w, stride_shape, padding=pad)

    if bias:
        b = tf.get_variable('b', [1, 1, 1, filters_num], initializer=tf.constant_initializer(0.0))
        conv = conv + b

    return conv


def conv2d_transpose(input, filters_num, filter_size, stride, reuse,
                     pad='SAME', dtype=tf.float32):
    assert pad == 'SAME'
    n, h, w, c = input.get_shape().as_list()
    stride_shape = [1, stride, stride, 1]
    filter_shape = [filter_size, filter_size, filters_num, c]
    output_shape = [n, h * stride, w * stride, filters_num]

    w = tf.get_variable('w', filter_shape, dtype, tf.initializers.random_normal(0.0, 0.02))
    deconv = tf.nn.conv2d_transpose(input, w, output_shape, stride_shape, pad)

    return deconv


def conv_block(input, filters_num, name, filter_size, stride, is_train, reuse, norm,
               activation, pad='SAME', bias=False):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d(input, filters_num, filter_size, stride, reuse, pad, bias=bias)
        out = norm_layer(out, is_train, reuse, norm)
        out = act_layer(out, activation)
        return out


def deconv_block(input, filters_num, name, k_size, stride, is_train, reuse, norm, activation):
    with tf.variable_scope(name, reuse=reuse):
        out = conv2d_transpose(input, filters_num, k_size, stride, reuse)
        out = norm_layer(out, is_train, reuse, norm)
        out = act_layer(out, activation)
        return out


def res_block(input, filters_num, name, is_train, reuse, norm, pad='REFLECT'):
    with tf.variable_scope(name, reuse=reuse):
        with tf.variable_scope('res1', reuse=reuse):
            out = conv2d(input, filters_num, 3, 1, reuse, pad)
            out = norm_layer(out, is_train, reuse, norm)
            out = act_layer(out, 'relu')

        with tf.variable_scope('res2', reuse=reuse):
            out = conv2d(out, filters_num, 3, 1, reuse, pad)
            out = norm_layer(out, is_train, reuse, norm)

        return act_layer(input + out, 'relu')
