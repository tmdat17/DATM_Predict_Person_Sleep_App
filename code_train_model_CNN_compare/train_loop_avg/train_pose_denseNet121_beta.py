# main train CNN (denseNet) POSE (stand, lie, sit)
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

from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from keras import layers
from keras import models

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

categories = ['lie', 'sit', 'stand', 'minus']

accuracyArr = []
precisionArr = []
recallArr = []
f1_scoreArr = []
timeTrainArr = []
timePredictArr = []

# -------------------------------------------------------------------------------
def accuracy(y_true, y_pred):
    correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    total = len(y_true)
    return correct / total if total != 0 else 0

def precision(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    predicted_positives = sum(1 for pred in y_pred if pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else 0

def recall(y_true, y_pred):
    true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == pred == 1)
    actual_positives = sum(1 for true in y_true if true == 1)
    return true_positives / actual_positives if actual_positives != 0 else 0

def f1_score(y_true, y_pred):
    precision_value = precision(y_true, y_pred)
    recall_value = recall(y_true, y_pred)
    return 2 * (precision_value * recall_value) / (precision_value + recall_value) if (precision_value + recall_value) != 0 else 0

def average(arr): 
    return sum(arr) / len(arr) 

i = 1
AVG = 6
time_full_avg_start = time.time()
while i <= AVG:
    print(f'LAN {i}')
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

    # plt.subplots(3,4)
    # for i in range(12):
    #     plt.subplot(3,4, i+1)
    #     plt.imshow(data[i], cmap="gray")
    #     plt.axis('off')
    #     plt.title(labels[i])
    # plt.show()

    print("train test split")
    # Chia dữ liệu thành tập train và test
    trainX, testX, trainY, testY = train_test_split(data, labels, test_size=0.2, random_state=42, shuffle=True)
    trainY = np_utils.to_categorical(trainY, len(categories))
    # Chia tập train thành tập train và validation
    trainX, valX, trainY, valY = train_test_split(trainX, trainY, test_size=0.2, random_state=42, shuffle=True)

    print("trainX shape: ", trainX.shape)
    print("testX shape: ", testX.shape)
    print("trainY shape: ", trainY.shape)
    print("testY shape: ", testY.shape)
    print("valX shape: ", valX.shape)
    print("valY shape: ", valY.shape)


    EPOCHS = 54
    INIT_LR = 1e-3
    BS = 64

    class_names = categories

    print("[INFO] compiling model...")
    DenseNet121 = DenseNet121(input_shape=(WIDTH, HEIGHT, 3), include_top=False, weights='imagenet')
    for layer in DenseNet121.layers:
        layer.trainable = False

    model = Sequential()
    model.add(DenseNet121)
    # model.add(layers.AveragePooling2D((8, 8), padding='valid', name='avg_pool'))
    model.add(GlobalAveragePooling2D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(len(class_names), activation='softmax'))
    print(model.summary())
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])


    print("bat dau fit model DenseNet121")
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=3)
    # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    start_time = time.time()
    history = model.fit(trainX, trainY, batch_size=BS, epochs=EPOCHS, validation_data=(valX, valY), verbose=1, callbacks=[early_stopping])
    end_time = time.time() - start_time
    timeTrainArr.append(end_time)

    print("new_model:  ", model)
    print("prepare save new_model:")
    # model.save("./model_denseNet121_train_pose/denseNet121_epo{}_bs{}_2500pics.h5".format(EPOCHS, BS))
    # Lấy các thông số từ đối tượng history
    loss_history = history.history['loss']
    val_loss_history = history.history['val_loss']
    accuracy_history = history.history['accuracy']
    val_accuracy_history = history.history['val_accuracy']

    # Vẽ biểu đồ loss và accuracy trên cùng một biểu đồ
    # plt.figure(figsize=(10, 6))
    # plt.plot(loss_history, color='red', linewidth=3, label='Training Loss')
    # plt.plot(val_loss_history, color='cyan', linewidth=3, label='Validation Loss')
    # plt.plot(accuracy_history, color='green', linewidth=3, label='Training Accuracy')
    # plt.plot(val_accuracy_history, color='blue', linewidth=3, label='Validation Accuracy')
    # plt.title('Model Loss and Accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Value')
    # plt.legend(loc='lower right')
    # plt.grid(True)
    # plt.savefig(f'./chart_loss_and_accuracy_denseNet121/epochs_{EPOCHS}_bs_{BS}_2500pics.png', dpi=300)
    # plt.show()


    print("bat dau kiem tra model: ")
    time_precdict = time.time()
    pred = model.predict(testX)
    end_time_predict = time.time() - time_precdict
    timePredictArr.append(end_time_predict)
    predictions = argmax(pred, axis=1) # return to label

    # cm = confusion_matrix(testY, predictions)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # cax = ax.matshow(cm)
    # plt.title('Model confusion matrix')
    # fig.colorbar(cax)
    # ax.set_xticklabels([''] + categories)
    # ax.set_yticklabels([''] + categories)

    # for i in range(len(class_names)):
    #     for j in range(len(class_names)):
    #         ax.text(i, j, cm[j, i], va='center', ha='center')

    # plt.xlabel('Predicted')
    # plt.ylabel('True')
    # plt.savefig(f'./confusion_matrix_denseNet121/epochs_{EPOCHS}_bs_{BS}_2500pics.png', dpi=300)
    # plt.show()


    # accuracy = accuracy_score(testY, predictions)
    # print("Accuracy : %.2f%%" % (accuracy*100.0))
    # print("\n")
    # # ----------------------------------------------

    # f1 = f1_score(testY, predictions,average='weighted')
    # print("F1 : %.2f%%" % (f1*100.0))
    # print("\n")

    # # ----------------------------------------------

    # recall= recall_score(testY, predictions,average='weighted')
    # print("Recall :%.2f%%" % (recall*100))
    # print("\n")


    # # ----------------------------------------------

    # precision = precision_score(testY, predictions,average='weighted')
    # print("Precision : %.2f%%" % (precision*100.0))
    # print("\n")


    # print('Time train DenseNet121: ', round(end_time/60,2))


    acc = accuracy(testY, predictions)
    prec = precision(testY, predictions)
    rec = recall(testY, predictions)
    f1 = f1_score(testY, predictions)
    
    print(f'accuracy {i}:  {acc * 100}')
    print(f'precision {i}:  {prec * 100}')
    print(f'recall {i}:  {rec * 100}')
    print(f'f1_score {i}:  {f1 * 100}')
    
    accuracyArr.append(acc * 100)
    precisionArr.append(prec * 100)
    recallArr.append(rec * 100)
    f1_scoreArr.append(f1 * 100)
    i += 1
    print('===========================================================================================')
time_full_avg_end = time.time() - time_full_avg_start

print(f'accuracyArr:  {accuracyArr}')
print(f'f1_scoreArr:  {f1_scoreArr}')
print(f'recallArr:  {recallArr}')
print(f'precisionArr:  {precisionArr}')
print(f'timeTrainArr:  {timeTrainArr}')
print(f'timePredictArr:  {timePredictArr}')


print(f'AVG {AVG} LAN    -    TOTAL TIME: {round(time_full_avg_end / 60,2)}')
print("Accuracy:", average(accuracyArr))
print("F1 Score:", average(f1_scoreArr))
print("Recall:", average(recallArr))
print("Precision:", average(precisionArr))
print('Time train DenseNet121: ', average(timeTrainArr))
print('Time predict DenseNet121: ', average(timePredictArr))