import os
import glob
import numpy as np
from PIL import Image
import shutil

data_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/data/"
train_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/dataset/train/"
test_dir = "/home/akshaj/projects_python/playground_py3.6/projects/flowers-keras-tf/dataset/test/"


classes = []

X_train = []
Y_train = []
X_test = []
Y_test = []

for name in os.listdir(data_dir):
    classes.append(name)

print(classes)

for i in classes:
    try:
        os.mkdir(train_dir + i)
        os.mkdir(test_dir + i)
    except:
        pass

# shutil.copyfile(data_dir + i + "/*", train_dir + j + "/")

for i in classes:
    count = 0
    for j in glob.glob(data_dir + i + "/*"):
        f = j.split('/')[-1]
        if count < 634:
            shutil.copy2(j, train_dir + i)
            # shutil.copyfile(j, train_dir + i + "/" + f)
        else:
            shutil.copy2(j, test_dir + i)
            # shutil.copyfile(j, test_dir + i + "/" + f)
        count += 1
