from os.path import join, abspath, dirname
import tensorflow as tf
from configparser import SafeConfigParser

# File hierarchy
_python_dir      = dirname(abspath(__file__))
_proyect_dir     = dirname(_python_dir)
_config_dir      = join(_proyect_dir, 'config')
_log_dir_root    = join(_proyect_dir, 'log')
_data_dir        = join(_proyect_dir, 'data')

# Configuration parser
cfg_parser = SafeConfigParser()
cfg_parser.read(join(_config_dir, 'cfg_model.ini'))

# Private variables
_initializer    = cfg_parser.get(section='train', option='initializer')
_normalization  = cfg_parser.get(section='train', option='normalization')
_batch_size     = cfg_parser.getint(section='train', option='batch_size')

def Xception(input_tensor, keep_prob, training):

    if _initializer == 'xavier':
        k_init = tf.contrib.layers.xavier_initializer()
    else:
        k_init = tf.random_normal_initializer(stddev=0.01)
    
    elu = tf.nn.elu

    def norm(x, norm_type, G=32, epsilon=1e-06):
        with tf.variable_scope('{}_norm'.format(norm_type)):
            if norm_type == 'none':
                output = x
            elif norm_type == 'batch':
                output = tf.contrib.layers.batch_norm(
                    inputs=x,
                    center=True,
                    scale=True,
                    decay=0.999,
                    is_training=training)
            elif norm_type == 'group':
                output = tf.contrib.layers.group_norm(
                    inputs=x,
                    groups=G,
                    channels_axis=-1,
                    reduction_axes=(-3, -2),
                    center=True,
                    scale=True,
                    epsilon=epsilon,
                    activation_fn=None,
                    param_initializers=None,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    scope=None)

            else:
                raise NotImplementedError
        return output
    
    def conv(inputs, filters, kernel_size=5, strides=1, name='conv', G=32):
        with tf.variable_scope(name) as scope:
            convol = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="VALID",
                activation=None,
                kernel_initializer=k_init,
                name='conv2d_'+name)
            convol = norm(convol, _normalization, G=G)
            return elu(convol, name = name+'_act')
    
    def sepconv(inputs, filters, kernel_size=5, strides=1, name='separable_conv2d', G=32):
        with tf.variable_scope(name) as scope:
            sepconvl = tf.layers.separable_conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding="VALID",
                name='sep_conv2d_'+name)
            sepconvl = norm(sepconvl, _normalization, G=G)
            return elu(sepconvl, name=name+'_act')
    
    conv1 = conv(input_tensor, filters=8, name='conv1', G=8)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max1')
    sepconv1 = sepconv(conv1, filters=16, name='sepconv1', G=8)
    sepconv1 = tf.nn.max_pool(sepconv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='max2')
    sepconv2 = sepconv(sepconv1, filters=120, name='sepconv2', G=8)
    
    with tf.variable_scope('flatten') as scope:
        fc0   = tf.contrib.layers.flatten(sepconv1)
        fc1   = tf.contrib.layers.flatten(sepconv2)
        concat_x = tf.concat([fc1, fc0], 1)
        concat_x = tf.nn.dropout(concat_x, keep_prob)
        nn_last_layer = tf.contrib.layers.fully_connected(concat_x, 43)

    return nn_last_layer