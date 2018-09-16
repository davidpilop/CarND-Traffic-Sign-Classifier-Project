import tensorflow as tf
from helper import conv

def LeNet4(input_tensor, keep_prob, training):
    conv1 = conv(input_tensor, training, filters=6, kernel_size=5, strides=1, padding='VALID', name='conv1', G=2)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='VALID', name='pool1')
    conv2 = conv(pool1, training, filters=16, kernel_size=5, strides=1, padding='VALID', name='conv2', G=4)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='VALID', name='pool2')
    conv3 = conv(pool2, training, filters=400, kernel_size=5, strides=1, padding='VALID', name='conv3', G=10)

    # Flatten. Input = 5x5x16. Output = 400.
    fc0   = tf.contrib.layers.flatten(pool2)
    # Flatten. Input = 5x5x16. Output = 400.
    fc1   = tf.contrib.layers.flatten(conv3)
    # Concat layer2flat and x. Input = 400 + 400. Output = 800
    concat_x = tf.concat([fc1, fc0], 1)
    # Dropout
    concat_x = tf.nn.dropout(concat_x, keep_prob)
    logits = tf.layers.dense(concat_x, units = 43, activation='elu')
    return logits