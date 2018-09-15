from os.path import abspath, dirname
from os.path import join as dir_join
import tensorflow as tf
from configparser import SafeConfigParser

# File hierarchy
_python_dir      = dirname(abspath(__file__))
_proyect_dir     = dirname(_python_dir)
_config_dir      = dir_join(_proyect_dir, 'config')
_log_dir_root    = dir_join(_proyect_dir, 'log')
_data_dir        = dir_join(_proyect_dir, 'data')

# Configuration parser
cfg_parser = SafeConfigParser()
cfg_parser.read(dir_join(_config_dir, 'cfg_model.ini'))

# Private variables
_initializer    = cfg_parser.get(section='train', option='initializer')
_normalization  = cfg_parser.get(section='train', option='normalization')
_batch_size     = cfg_parser.getint(section='train', option='batch_size')

def safe_div(numerator, denominator, name='safe_div'):
    """Divides two values, returning 0 if the denominator is <= 0.

    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.

    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(tf.greater(denominator, 0),
                    tf.truediv(numerator, denominator),
                    0,
                    name=name)

def metrics(nn_last_layer, correct_label):
    Intersection_Total = []
    # NumelLabels_Total = []
    NumelPreds_Total = []
    # Union_Total = []

    with tf.name_scope("metrics") as scope:
        for cls_id in range(1, 43): # TODO: NUM_CLASSES
            # IoU
            labels_cls = tf.cast(tf.equal(correct_label, cls_id),dtype=tf.float32)
            preds_cls = tf.cast(tf.equal(tf.argmax(nn_last_layer, -1), cls_id),dtype=tf.float32)
            intersection = tf.reduce_sum(tf.multiply(labels_cls,preds_cls))
            # numelLabels = tf.reduce_sum(labels_cls)
            numelPreds = tf.reduce_sum(preds_cls)
            # union = tf.subtract(tf.add(numelLabels, numelPreds),intersection)
            # IoU = safe_div(intersection,union)
            # tf.summary.scalar('iou/' + CLASS2LABEL[cls_id], IoU)

            # Recall
            # recall = safe_div(intersection, numelLabels)
            # tf.summary.scalar('recall/' + CLASS2LABEL[cls_id], recall)

            # Accuracy
            accuracy = safe_div(intersection, numelPreds)
            tf.summary.scalar('accuracy_' + str(cls_id), accuracy)# TODO:CLASS2LABEL

            Intersection_Total.append(intersection)
            # NumelLabels_Total.append(numelLabels)
            NumelPreds_Total.append(numelPreds)
            # Union_Total.append(union)

    # return Intersection_Total, NumelLabels_Total, NumelPreds_Total, Union_Total
    return Intersection_Total, NumelPreds_Total


elu = tf.nn.elu

def norm(x, norm_type, training, G=32, epsilon=1e-06):
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
    
def conv(inputs, training, filters, kernel_size=3, strides=1, padding='SAME', name='conv', G=32):
    if _initializer == 'xavier':
        k_init = tf.contrib.layers.xavier_initializer()
    else:
        k_init = tf.random_normal_initializer(stddev=0.01)

    with tf.variable_scope(name) as scope:
        convol = tf.layers.conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None,
            kernel_initializer=k_init,
            name='conv2d_'+name)
        convol = norm(convol, _normalization, training, G=G)
        return elu(convol, name = name+'_act')
    
def sepconv(inputs, training, filters, kernel_size=3, strides=2, padding='SAME', name='separable_conv2d', G=32):
    if _initializer == 'xavier':
        k_init = tf.contrib.layers.xavier_initializer()
    else:
        k_init = tf.random_normal_initializer(stddev=0.01)

    with tf.variable_scope(name) as scope:
        sepconvl = tf.layers.separable_conv2d(
            inputs=inputs,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depthwise_initializer=k_init,
            pointwise_initializer=k_init,
            name='sep_conv2d_'+name)
        sepconvl = norm(sepconvl, _normalization, training, G=G)
        return elu(sepconvl, name=name+'_act')