# Load libraries
import pickle
import csv
import glob
import sys, select
import os
from os import mkdir
from os.path import join, dirname, exists
from shutil import copyfile
from datetime import datetime

from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import random
from sklearn.utils import shuffle
from configparser import SafeConfigParser

import tensorflow as tf

from prepare_data import preprocess
from LeNet import LeNet
from LeNet2 import LeNet2
from Xception import Xception

# File hierarchy
_python_dir      = dirname(os.path.abspath(__file__))
_proyect_dir     = dirname(_python_dir)
_config_dir      = join(_proyect_dir, 'config')
_log_dir_root    = join(_proyect_dir, 'log')
_data_dir        = join(_proyect_dir, 'data')

# Configuration parser
cfg_parser = SafeConfigParser()
cfg_parser.read(join(_config_dir,'cfg_model.ini'))

# Private variables
_pretrained            = cfg_parser.getboolean(section='pretrained_model', option='pretrained')
_pretrained_model_path = cfg_parser.get(section='pretrained_model', option='pretrained_model_path')

_models_group_dir      = cfg_parser.get(section='model_saving', option='models_group_dir')
_model_log_dir         = cfg_parser.get(section='model_saving', option='model_log_dir')

_max_epoch             = cfg_parser.getint(section='train', option='max_epoch')
_base_learning_rate    = cfg_parser.getfloat(section='train', option='base_learning_rate')
_decay_step            = cfg_parser.getint(section='train', option='decay_step')
_decay_rate            = cfg_parser.getfloat(section='train', option='decay_rate')
_keep_prob             = cfg_parser.getfloat(section='train', option='keep_prob')
_batch_size            = cfg_parser.getint(section='train', option='batch_size')

_training_file         = join(_data_dir, cfg_parser.get(section='data_set', option='training_file'))
_validation_file       = join(_data_dir, cfg_parser.get(section='data_set', option='validation_file'))
_testing_file          = join(_data_dir, cfg_parser.get(section='data_set', option='testing_file'))

# Set log and pretrained model direcotires
_current_log_dir = join(_log_dir_root, _models_group_dir)
if not exists(_current_log_dir): mkdir(_current_log_dir)
_current_log_dir = join(_current_log_dir, _model_log_dir)
if exists(_current_log_dir):
    print("Do you want to overwrite the log? [y,n]: ")
    while True:
        i, o, e = select.select([sys.stdin], [], [], 10)
        if(i):
            confirm = sys.stdin.readline().strip()
            if confirm == 'y':
                os.system('rm -rf ' + _current_log_dir)
                break
            elif confirm == 'n':
                _current_log_dir = _current_log_dir + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
                break
            else:
                print("\n Invalid Option. Please Enter a Valid Option [y,n]: ")
        else:
            _current_log_dir = _current_log_dir + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')
            break
mkdir(_current_log_dir)
_current_log_dir_files = join(_current_log_dir, 'files')
if not exists(_current_log_dir_files): mkdir(_current_log_dir_files)

_out_log_file = open(join(_current_log_dir, 'trainning_results.txt'), 'a')

os.system('cp ' + join(_python_dir,'LeNet.py') + ' %s' % (_current_log_dir_files))
os.system('cp ' + join(_python_dir,'train.py') + ' %s' % (_current_log_dir_files))
os.system('cp ' + join(_config_dir,'cfg_model.ini') + ' %s' % (_current_log_dir_files))

with open(_training_file, mode='rb') as f:
    trainData = pickle.load(f)
with open(_validation_file, mode='rb') as f:
    validData = pickle.load(f)
with open(_testing_file, mode='rb') as f:
    testData = pickle.load(f)
        
X_train, y_train = trainData['features'], trainData['labels']
X_valid, y_valid = validData['features'], validData['labels']
X_test, y_test = testData['features'], testData['labels']
n_train = len(X_train)              # Number of training examples
n_valid = len(X_valid)                # Number of testing examples.
n_test = len(X_test)                # Number of testing examples.
image_shape = X_train[0].shape      # What's the shape of an traffic sign image?
n_classes = len(np.unique(y_train)) # How many unique classes/labels there are in the dataset.

print("X_train shape:", X_train.shape)
print("X_valid shape:", X_valid.shape)
print("X_test shape:", X_test.shape)
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

X_train_normalize = preprocess(X_train)
X_valid_normalize = preprocess(X_valid)
X_test_normalize = preprocess(X_test)


# Write to file function
def log_string(out_str):
    _out_log_file.write(out_str+'\n')
    _out_log_file.flush()

def placeholder():
    x = tf.placeholder(tf.float32, shape=(_batch_size, 32, 32, 1), name='input')
    y = tf.placeholder(tf.int32, shape=_batch_size, name='labels')
    keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob') # probability to keep units
    training = tf.placeholder(tf.bool, shape=None, name='keep_prob')
    one_hot_y = tf.one_hot(y, 43, name='one_hot_labels') # TODO: num_classes from signnames.csv
    return x, y, keep_prob, one_hot_y, training

def optimize(logits, one_hot_y, global_step):
    with tf.variable_scope('optimize') as scope:
        # Calculating Loss
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits),name='loss')
        tf.summary.scalar('loss', cross_entropy_loss)
        # Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate = _base_learning_rate)
        train_op = optimizer.minimize(cross_entropy_loss, global_step=global_step)
        # tf.summary.scalar('learning_rate', train_op._lr_t)
    return train_op

def train_one_epoch(sess, ops, writer, epoch, X_data, y_data):
    X_data, y_data = shuffle(X_data, y_data)
    for offset in tqdm(range(0, len(X_data), _batch_size), ncols= 100, desc = ("EPOCH {} ".format(epoch+1))):
        end = offset + _batch_size
        if end > n_train:
            batch_x, batch_y = np.asarray(X_data[n_train-_batch_size:n_train]), np.asarray(y_data[n_train-_batch_size:n_train])
        else:
            batch_x, batch_y = np.asarray(X_data[offset:end]), np.asarray(y_data[offset:end])

        feed_dict={ops['x']: batch_x,
                   ops['y']: batch_y,
                   ops['keep_prob']: _keep_prob,
                   ops['training']: True}
        _, summary, step = sess.run([ops['train_op'],
                                     ops['merged'],
                                     ops['step']],
                                     feed_dict=feed_dict)
        writer.add_summary(summary, step)
    return

def eval_one_epoch(sess, ops, writer, epoch, X_data, y_data):
    for offset in range(0, len(X_data), _batch_size):
        end = offset + _batch_size
        if end > n_valid:
            batch_x, batch_y = np.asarray(X_data[n_valid-_batch_size:n_valid]), np.asarray(y_data[n_valid-_batch_size:n_valid])
        else:
            batch_x, batch_y = np.asarray(X_data[offset:end]), np.asarray(y_data[offset:end])

        feed_dict={ops['x']: batch_x,
                   ops['y']: batch_y,
                   ops['keep_prob']: 1.0,
                   ops['training']: False}
        _, summary, step = sess.run([ops['accuracy'],
                                     ops['merged'],
                                     ops['step']],
                                     feed_dict=feed_dict)
        writer.add_summary(summary, step)
    return

def metrics(logits, one_hot_y):
    '''
    Accuracy of a given set of predictions of size (N x n_classes) and
    labels of size (N x n_classes)
    '''
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1)), tf.float32)
    accuracy_operation = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy_operation)
    tf.metrics.mean_per_class_accuracy(
        labels = one_hot_y,
        predictions = logits,
        num_classes = 43,
        weights=None,
        metrics_collections=None,
        updates_collections=None,
        name='accuracy_per_class')
    return accuracy_operation

def train():
    # Create a session
    sess = tf.Session(graph=tf.get_default_graph())

    x, y, keep_prob, one_hot_y, training = placeholder()
    logits = LeNet2(x, keep_prob)
    accuracyTotal = metrics(logits, one_hot_y)

    global_step = tf.Variable(0, trainable=False)
    train_op = optimize(logits, one_hot_y, global_step)

    # Log information
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(join(_current_log_dir, 'train'), sess.graph)
    valid_writer = tf.summary.FileWriter(join(_current_log_dir, 'valid'))

    # If model starts from previous training:
    if _pretrained:
        assert exists(_pretrained_model_path), "Pretrained model doesn't found"
        check = tf.train.latest_checkpoint(_pretrained_model_path)
        saver.restore(sess, check)
        log_string("Model restored.")
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    # Define TensorFlow operations
    ops = {'x': x,
           'y': y,
           'logits': logits,
           'train_op': train_op,
           'merged': merged,
           'step': global_step,
           'keep_prob': keep_prob,
           'training': training,
           'accuracy': accuracyTotal}

    # Train and eval each epoch
    for epoch in range(_max_epoch):
        train_one_epoch(sess, ops, train_writer, epoch, X_train_normalize, y_train)
        eval_one_epoch(sess, ops, valid_writer, epoch, X_valid_normalize, y_valid)

        # Save model chekpoint to disk
        if epoch % 10 == 0:
            save_path = saver.save(sess, join(_current_log_dir_files, "model"))

    # Save final state
    save_path = saver.save(sess, join(_current_log_dir_files, "model"))
    log_string("\nModel saved in file: %s" % save_path)

if __name__ == '__main__':
    train()
    _out_log_file.close()
    copyfile(join(_current_log_dir, 'trainning_results.txt'), join(_proyect_dir,'trainning_results.txt'))
