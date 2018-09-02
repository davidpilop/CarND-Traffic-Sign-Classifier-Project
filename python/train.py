# Load libraries
import pickle
import csv
import glob
import os

from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import random
from sklearn.utils import shuffle

import tensorflow as tf

from prepare_data import preprocess
from LeNet import LeNet
from LeNet2 import LeNet2

#Import data
training_file = './data/train.p'
validation_file = './data/valid.p'
testing_file = './data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
        
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']
n_train = len(X_train)              # Number of training examples
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


EPOCHS = 60
BATCH_SIZE = 2047
rate = 0.0009
num_classes = 43 # TODO: read signnames.csv

def placeholder():
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 1), name='input')
    y = tf.placeholder(tf.int32, shape=None, name='labels')
    keep_prob = tf.placeholder(tf.float32, shape=None, name='keep_prob') # probability to keep units
    one_hot_y = tf.one_hot(y, num_classes, name='one_hot_labels')
    return x, y, keep_prob, one_hot_y

def main_optimizer(logits, one_hot_y):
    # Calculating Loss
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits),name='loss')
    tf.summary.scalar('loss', tf_loss)
    # Optimizer
    tf_loss_minimize = tf.train.AdamOptimizer(learning_rate = rate).minimize(tf_loss)    
    return tf_loss_minimize

def evaluate_one_epoch(X_data, y_data, ops):
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, len(X_data), BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        feed_dict = {ops['x']: batch_x, ops['y']: batch_y, ops['keep_prob']: 1.0}
        accuracy = sess.run(ops['accuracy_operation'], feed_dict=feed_dict)
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / len(X_data)

def accuracy(logits, one_hot_y):
    '''
    Accuracy of a given set of predictions of size (N x n_classes) and
    labels of size (N x n_classes)
    '''
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1)), tf.float32)
    accuracy_operation = tf.reduce_mean(correct_prediction)
    tf.summary.scalar('accuracy', accuracy_operation)
    return accuracy_operation

def train_one_epoch(sess, X_data, y_data, ops, epoch):
    X_data, y_data = shuffle(X_data, y_data)
    for offset in tqdm(range(0, len(X_data), BATCH_SIZE), ncols= 100, desc = ("EPOCH {} ".format(epoch+1))):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        feed_dict={ops['x']: batch_x, ops['y']: batch_y, ops['keep_prob']: 0.5}
        summary, _ = sess.run([ops['merge'], ops['tf_loss_minimize']], feed_dict=feed_dict)
        return summary

def main_train(X_train_normalize, y_train):
    tf.reset_default_graph()
    
    x, y, keep_prob, one_hot_y = placeholder()
    logits = LeNet2(x, keep_prob)
    tf_loss_minimize = main_optimizer(logits, one_hot_y)

    accuracy_operation = accuracy(logits, one_hot_y)
    saver = tf.train.Saver()

    merge = tf.summary.merge_all()

    ops = {
        'x': x,
        'y': y,
        'keep_prob': keep_prob,
        'one_hot_y': one_hot_y,
        'logits': logits,
        'tf_loss_minimize': tf_loss_minimize,
        'accuracy_operation': accuracy_operation,
        'merge': merge}

    with tf.Session() as sess:
        summ_writer = tf.summary.FileWriter(os.path.join('log'), sess.graph)

        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init)
        
        print("Training...\n")
        for epoch in range(EPOCHS):
            summary = train_one_epoch(sess, X_train_normalize, y_train, ops, epoch)
            summ_writer.add_summary(summary, epoch)                
            validation_accuracy = evaluate_one_epoch(X_valid_normalize, y_valid, ops)
            
        saver.save(sess, './lenet')
        print("Model saved")
        summ_writer.close()

if __name__ == "__main__":
    main_train(X_train_normalize, y_train)