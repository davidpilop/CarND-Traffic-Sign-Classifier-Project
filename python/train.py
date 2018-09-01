# Load libraries
import pickle
import csv
import glob

from tqdm import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import numpy as np
import cv2
import random

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

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
BATCH_SIZE = 100
rate = 0.0009

def placeholder():
    x = tf.placeholder(tf.float32, (None, 32, 32, 1), name='input')
    y = tf.placeholder(tf.int32, (None), name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob') # probability to keep units
    one_hot_y = tf.one_hot(y, 43, name='one_hot_labels')
    return x, y, keep_prob, one_hot_y

def main_optimizer(logits, one_hot_y):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate = rate).minimize(loss_operation)
    return train_op

def main_train():
    tf.reset_default_graph()

    x, y, keep_prob, one_hot_y = placeholder()
    logits = LeNet2(x, keep_prob)
    train_op = main_optimizer(logits, one_hot_y)
