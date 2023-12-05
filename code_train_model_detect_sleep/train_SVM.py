import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from tensorflow.keras import optimizers
from sklearn.svm import SVC,SVR
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


path="./luon/"
categories = ['BD_KT','DT_mu', 'luon_1', 'luon_2','luon_3']

data = []#dữ liệu
labels = []#nhãn
imagePaths = []
HEIGHT = 32
WIDTH = 32
# 24 24
N_CHANNELS = 3

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    label = imagePath[1]
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

EPOCHS = 5
INIT_LR = 1e-3
BS =32

class_names = categories


model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(strides=2))
model.add(Convolution2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))
print(model.summary())

opt=tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)
print("bat dau tric dat trung")
new_model=Model(inputs=model.input, outputs=model.get_layer('dense_1').output)
print("new_model: ", new_model)

feat_train=new_model.predict(trainX)
print("feat_train",feat_train.shape)

feat_test=new_model.predict(testX)
print("feat_test",feat_test.shape)

print (" khoi tao SVR")
model_SVM=SVC(kernel="rbf", C=10000, gamma=0.001)
model_SVM.fit(feat_train,np.argmax(trainY,axis=1))
prepY=model_SVM.predict(feat_test)
# print("day ket qua dat trung",prepY)

accuracy= accuracy_score(testY, prepY,average='weighted')
print("F1 : %.2f%%" % (accuracy *100.0))
print("\n")

recall= recall_score(testY, prepY,average='weighted')
print("F1 : %.2f%%" % (accuracy *100.0))
print("\n")

precision= precision_score(testY, prepY,average='weighted')
print("F1 : %.2f%%" % (accuracy *100.0))
print("\n")

f1 = f1_score(testY, prepY,average='weighted')
print("F1 : %.2f%%" % (f1*100.0))
print("\n")


