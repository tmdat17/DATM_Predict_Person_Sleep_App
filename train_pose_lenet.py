# code sample CNN (lenet)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras

# ========================duong dan======================================

# path="./vtho/"
# categories = ['BD_KT','DT_mu','vtho_1']

# path="./tay/"
# categories = ['BD_KT','DT_mu', 'tay_1', 'tay_2','tay_3']

# path="./chan/"
# categories = ['BD_KT', 'chan_1', 'chan_2','DT_mu']

path="./luon/"
categories = ['BD_KT','DT_mu', 'luon_1', 'luon_2','luon_3']

# path="./bung/"
# categories = ['BD_KT', 'bung_1', 'bung_2','bung_3','DT_mu']

# path="./toanthan/"
# categories = ['BD_KT','DT_mu','toanthan_1', 'toanthan_2','toanthan_3']

# path="./nhay/"
# categories = ['BD_KT','DT_mu', 'nhay_1', 'nhay_2']

# path="./dieuhoa/"
# categories = ['BD_KT', 'dieuhoa_1', 'dieuhoa_2','DT_mu']

# =============================resize kich thuoc anh=================================

data = []#dữ liệu
labels = []#nhãn
imagePaths = []
HEIGHT = 64
WIDTH = 64
# 24 24
N_CHANNELS = 3

# ===========================lay ngau nhien anh===================================

for k, category in enumerate(categories):
    for f in os.listdir(path+category):
        imagePaths.append([path+category+'/'+f, k]) 

import random
random.shuffle(imagePaths)
# print(imagePaths[:10])

# =======================tien xu ly=======================================

for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))  # .flatten()
    data.append(image)
    label = imagePath[1]
    labels.append(label)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

plt.subplots(3,4)
for i in range(12):
    plt.subplot(3,4, i+1)
    plt.imshow(data[i])
    plt.axis('off')
    plt.title(categories[labels[i]])
# plt.show()

# ============================chia tap dl==================================

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)# random_state=30)

trainY = np_utils.to_categorical(trainY, len(categories))

# ===========================huan luyen===================================

EPOCHS = 10
INIT_LR = 1e-3
BS =100

class_names = categories


model = Sequential()

model.add(Convolution2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D(strides=2))
model.add(Convolution2D(filters=48, kernel_size=(5,5), padding='valid', activation='relu'))
model.add(MaxPooling2D(strides=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(len(class_names), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, verbose=1)

# model.save("vuontho100.h5")
# ==========================kiem tra su dung cua mo hinh====================================

from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

pred = model.predict(testX)
predictions = argmax(pred, axis=1) # return to label

cm = confusion_matrix(testY, predictions)

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title('Model confusion matrix')
fig.colorbar(cax)
ax.set_xticklabels([''] + categories)
ax.set_yticklabels([''] + categories)

for i in range(len(class_names)):
    for j in range(len(class_names)):
        ax.text(i, j, cm[j, i], va='center', ha='center')

plt.xlabel('Predicted')
plt.ylabel('True')
# plt.show()


accuracy = accuracy_score(testY, predictions)
print("Accuracy : %.2f%%" % (accuracy*100.0))
print("\n")
# ----------------------------------------------

recall= recall_score(testY, predictions,average='weighted')
print("Recall :%.2f%%" % (recall*100))
print("\n")
# ----------------------------------------------

precision = precision_score(testY, predictions,average='weighted')
print("Precision : %.2f%%" % (precision*100.0))
print("\n")
# ----------------------------------------------

f1 = f1_score(testY, predictions,average='weighted')
print("F1 : %.2f%%" % (f1*100.0))
print("\n")

# ==============================dua anh vao kiem tra================================

# from numpy import argmax
# import PIL
# import matplotlib.pyplot as plt
# from tensorflow.keras.preprocessing import image
# img_path="./nhan480.jpg"

# img=image.load_img(img_path,target_size=(32,32))
# img_array=image.img_to_array(img)
# from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
# img_batch=np.expand_dims(img_array, axis=0)
# img_preprocessed=preprocess_input(img_batch)

# pred=model.predict(img_preprocessed)
# Res=argmax(pred,axis=1)
# print(pred)

# plt.imshow(img)
# plt.show()
# print(categories[Res[0]],pred[0][Res[0]]*100)