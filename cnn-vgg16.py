import os
import glob
import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.layers import Dropout
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.applications import VGG16
import h5py

train_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/dataset/train/"
test_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/dataset/test/"

classes = []
X_train = []
Y_train = []
X_test = []
Y_test = []

for name in os.listdir(train_dir):
    classes.append(name)

print("classes: ", classes)


for j in classes:
    for trdata in glob.glob(train_dir + j + "/*.jpg"):
        print(trdata)
        image = cv2.imread(trdata)
        image = cv2.resize(image, (128, 128))
        X_train.append(image)
        Y_train.append(j)

    for trdata in glob.glob(test_dir + j + "/*.jpg"):
        image = cv2.imread(trdata)
        image = cv2.resize(image, (128, 128))
        X_test.append(image)
        Y_test.append(j)

X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_train = X_train / 255
X_test = X_test / 255

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


le = LabelEncoder()
Y_test = le.fit_transform(Y_test)
Y_train = le.fit_transform(Y_train)


Y_test = to_categorical(Y_test)
Y_train = to_categorical(Y_train)

print(Y_train.shape)
print(Y_test.shape)


img_width = 128
img_height = 128
nb_train_samples = 3710
nb_test_samples = 1152
batch_size = 128
epochs = 25
num_of_classes = 5

model = VGG16(weights='imagenet',
              include_top=False,
              input_shape=(128, 128, 3))

for layer in model.layers[:-5]:
    layer.trainable = False

top_layers = model.output
top_layers = Flatten(input_shape=model.output_shape[1:])(top_layers)
top_layers = Dense(num_of_classes, activation="relu",
                   input_shape=(num_of_classes,))(top_layers)
top_layers = Dropout(0.5)(top_layers)


# top_layers = Dense(num_of_classes, activation="relu",input_shape=(num_of_classes,))(top_layers)
top_layers = Dense(num_of_classes, activation="softmax")(top_layers)


# Add top layers on top of freezed (not re-trained) layers of VGG16
model_final = Model(inputs=model.input, outputs=top_layers)

model_final.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

history = model_final.fit(X_train, Y_train,
                          batch_size=64, epochs=25, verbose=1)


score = model.evaluate(X_test, Y_test, verbose=0)

print("Loss = " + str(score[0]))
print("Test Accuracy = " + str(score[1]))

model.save("flower-vgg16.h5py")
