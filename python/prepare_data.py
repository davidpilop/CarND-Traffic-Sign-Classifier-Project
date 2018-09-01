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

# Visualizations
def visualization(n_classes, X_train):
    plt.figure(figsize=(15, 15))
    for i in range(0, n_classes):
        plt.subplot(7, 7, i+1)
        x_selected = X_train[y_train == i]
        plt.imshow(x_selected[0, :, :, :])
        plt.title(i)
        plt.axis('off')
    plt.show()

#Plot number of images per class
def histogram():
    plt.figure(figsize=(12, 4))
    plt.bar(range(0, n_classes), np.bincount(y_train))
    plt.title("Distribution of the train dataset")
    plt.xlabel("Class number")
    plt.ylabel("Number of images")
    plt.show()

    print("Min number of images per class =", min(np.bincount(y_train)))
    print("Max number of images per class =", max(np.bincount(y_train)))

def preprocess(data):
    data_gry = np.sum(data/3, axis=3, keepdims = True)
    return (data_gry - 128)/128

def random_translate(img):
    px = 2
    dx,dy = np.random.randint(-px,px,2)
    M = np.float32([[1,0,dx],[0,1,dy]])
    return cv2.warpAffine(img,M,img.shape[:2])

def random_scaling(img):
    rows,cols = img.shape
    px = np.random.randint(-2,2)
    pts1 = np.float32([[px,px],[rows-px,px],[px,cols-px],[rows-px,cols-px]])
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    return cv2.warpPerspective(img,M,img.shape[:2])

def rotate_img(img):
    c_x,c_y = int(img.shape[0]/2), int(img.shape[1]/2)
    ang = 30.0*np.random.rand()-15
    Mat = cv2.getRotationMatrix2D((c_x, c_y), ang, 1.0)
    return cv2.warpAffine(img, Mat, img.shape[:2])

#Compute linear image transformation img*s+m
def lin_img(img,s=1.0,m=0.0):
    img2 = cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))

#Change image contrast; s>1 - increase
def contr_img(img, s=1.0):
    m = 127.0*(1.0-s)
    return lin_img(img, s, m)

def augment(img):
    img = contr_img(img, 1.8*np.random.rand()+0.2)
    img = rotate_img(img)
    img = random_translate(img)
    return random_scaling(img)


def dataAugmentation(X_train_normalize, y_train):
    print('X, y shapes:', X_train_normalize.shape, y_train.shape)

    for class_n in range(n_classes):
        class_indices = np.where(y_train == class_n)
        n_samples = len(class_indices[0])
        if n_samples < 800:
            for i in tqdm(range(800 - n_samples), ncols= 100, ascii = True, desc = str(class_n)):
                new_img = augment(X_train_normalize[class_indices[0][i % n_samples]])[:,:,np.newaxis]
                X_train_normalize = np.concatenate((X_train_normalize, [new_img]), axis=0)
                y_train = np.concatenate((y_train, [class_n]), axis=0)
                
    print('X, y shapes:', X_train_normalize.shape, y_train.shape)
    return X_train_normalize, y_train