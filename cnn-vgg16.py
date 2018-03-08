import os
import glob
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.utils import np_utils

train_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/dataset/train/"
test_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/dataset/test/"

classes = []
X_train = []
Y_train = []
X_test = []
Y_test = []

for name in os.listdir(train_dir):
    classes.append(name)

print(classes)


for j in classes:
    for trdata in glob.glob(train_dir + j + "/*.jpg"):
        image = cv2.imread(trdata)
        X_train.append(image)
        Y_train.append(j)

    for trdata in glob.glob(test_dir + j + "/*.jpg"):
        image = cv2.imread(trdata)
        X_test.append(image)
        Y_test.append(j)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = X_train.astype("float64") / 255
X_test = X_test / 255

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

Y_test = to_categorical(Y_test)
Y_train = to_categorical(Y_train)

print(Y_test.shape)
print(Y_train.shape)
