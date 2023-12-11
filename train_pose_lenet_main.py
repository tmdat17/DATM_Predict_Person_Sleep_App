# main train CNN (lenet) POSE (stand, lie, sit)
import numpy as np 
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
from scipy import stats
import os
import cv2
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
# from tensorflow.keras import optimizers
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
from keras.callbacks import EarlyStopping
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score


categories = ['lie', 'sit', 'stand', 'minus']

# path = "./data_detected/"
path = "./data_cropped/at_home/"

data = []
labels = []
imagePaths = []
WIDTH=128
HEIGHT=128
N_CHANNELS = 3

for k, category in enumerate(categories):
  for f in os.listdir(path+category):
    imagePaths.append([path+category+'/'+f,k]) # k=0: '01', k=1: '03', k=2: '05'

import random
random.shuffle(imagePaths)
print(imagePaths[:10])

print("Chuan bi doc anh tu folder: ")
for imagePath in imagePaths:
    image = cv2.imread(imagePath[0])
    image = cv2.resize(image, (WIDTH, HEIGHT))
    # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data.append(image)
    label = imagePath[1]
    labels.append(label)
# scale the raw pixel intensities to the range [0, 1]
print("scale raw pixel / 255.0")
data = np.array(data, dtype="float32") / 255.0
labels = np.array(labels)

plt.subplots(3,4)
for i in range(12):
    plt.subplot(3,4, i+1)
    plt.imshow(data[i], cmap="gray")
    plt.axis('off')
    plt.title(labels[i])
plt.show()

print("train test split")
# Chia dữ liệu thành tập train và test
trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42)
trainY = np_utils.to_categorical(trainY, len(categories))
# Chia tập train thành tập train và validation
trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42)

print("trainX shape: ", trainX.shape)
print("testX shape: ", testX.shape)
print("trainY shape: ", trainY.shape)
print("testY shape: ", testY.shape)
print("valX shape: ", valX.shape)
print("valY shape: ", valY.shape)


EPOCHS = 20
INIT_LR = 1e-3
BS = 5
print("[INFO] compiling model...")

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
print(model.summary())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


print("bat dau fit model")
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
# early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, validation_data=(valX, valY), verbose=1, callbacks=[early_stopping])

print("new_model:  ", model)
print("prepare save new_model:")
model.save("./model_lenet_train_pose/lenet_epo{}_bs{}.h5".format(EPOCHS, BS))
# new_model.save('./model_extract_feature_CNN/lenet_extract_features_epo{}_bs{}.h5'.format(EPOCHS, BS))
# Lấy các thông số từ đối tượng history
loss_history = history.history['loss']
val_loss_history = history.history['val_loss']
accuracy_history = history.history['accuracy']
val_accuracy_history = history.history['val_accuracy']

# Vẽ biểu đồ loss và accuracy trên cùng một biểu đồ
plt.figure(figsize=(10, 6))
plt.plot(loss_history, color='red', linewidth=3, label='Training Loss')
plt.plot(val_loss_history, color='cyan', linewidth=3, label='Validation Loss')
plt.plot(accuracy_history, color='green', linewidth=3, label='Training Accuracy')
plt.plot(val_accuracy_history, color='blue', linewidth=3, label='Validation Accuracy')
plt.title('Model Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(f'./chart_loss_and_accuracy_lenet/epochs_{EPOCHS}_bs_{BS}.png', dpi=300)
plt.show()


print("bat dau kiem tra model: ")
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
plt.show()
plt.savefig(f'./confusion_matrix_lenet/epochs_{EPOCHS}_bs_{BS}.png', dpi=300)


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