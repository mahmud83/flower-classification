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

X_train = X_train / 255
X_test = X_test / 255

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

Y_test = to_categorical(Y_test)
Y_train = to_categorical(Y_train)

print(Y_test.shape)
print(Y_train.shape)


model = Sequential()
model.add(Convolution2D(32, (3, 3), input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))


model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 9. Fit model on training data
start = time.time()
model_info = model.fit(X_train, Y_train,
                       batch_size=64, epochs=25, verbose=1)
end = time.time()
# plot model history
plot_model_history(model_info)
print(f"Model took {start} - {end} time to train.")

# 10. Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print("Loss = " + str(score[0]))
print("Test Accuracy = " + str(score[1]))

model.save(
    "/home/akshaj/projects_python/playground_py3.6/projects/cifar10-keras-tf/cifar-model.h5py")
