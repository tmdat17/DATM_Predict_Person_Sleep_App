import time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import binarize

# Generate synthetic data for binary classification
# Doc du lieu tu file
test_accuracyArr = []
test_f1_scoreArr = []
test_recallArr = []
test_precisionArr = []
TOTAL_TIME = []
i = 0
AVG = 50
K = 10
LIMIT_LINE = 3
EPOCH = 20
BS = 24
while i < AVG:
    dulieuload = pd.read_csv(f"./data_train_text/{LIMIT_LINE}_line_merge_2_lie_to_train.txt", delimiter=' ')
    dulieuload.replace({'s': 1, 'w': 0}, inplace=True)

    # print(dulieuload)
    dulieu_x = dulieuload.iloc[:, 0:-1]
    # Doc cot label 
    dulieu_y = dulieuload.iloc[:, -1]
    # print('label:  ', dulieu_y)

    print('len full data: ', len(dulieuload))
    print('len x data: ', len(dulieu_x))
    print('len y data: ', len(dulieu_y))

    
    # ----------------------------
    # Chia tap du lieu
    # Danh gia nghi thuc K-fold
    kf = KFold(n_splits=K, shuffle=True)
    for idtrain, idtest in kf.split(dulieuload):
        x_train = dulieu_x.iloc[idtrain,]
        x_test = dulieu_x.iloc[idtest,]
        y_train = dulieu_y.iloc[idtrain]
        y_test = dulieu_y.iloc[idtest]

    print("6 dong dau x_train: \n", x_train)
    print("6 dong dau x_test: \n", x_test)
    # Build the 1D CNN model
    # model = keras.Sequential([
    #     keras.layers.Input(shape=(12, 1)),  # Input shape (100 features, 1 channel)
    #     keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #     keras.layers.MaxPooling1D(pool_size=2),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(64, activation='relu'),
    #     keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    # ])


    # checked ok
    model = keras.Sequential([
        keras.layers.Input(shape=(LIMIT_LINE * 4, 1)),  # Input shape (100 features, 1 channel)
        keras.layers.Conv1D(32, kernel_size=5, activation='relu'),
        keras.layers.MaxPooling1D(pool_size=2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])


    # model = keras.Sequential([
    #     keras.layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=(12, 1)),
    #     keras.layers.MaxPooling1D(pool_size=2),
    #     keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
    #     keras.layers.MaxPooling1D(pool_size=2),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(128, activation='relu'),
    #     keras.layers.Dense(2, activation='sigmoid')
    # ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    print(f'LAN: {i+1}')
    
    # Train the model
    each_time_cur = time.time()
    model.fit(x_train, y_train, epochs=EPOCH, batch_size=BS, validation_split=0.2)
    end_each_time = time.time() - each_time_cur
    print(f'time train lan {i}:  ', end_each_time)
    TOTAL_TIME.append(end_each_time)

    # Evaluate the model on the test data
    print('---------------------------------------------------------------------------------')
    print(x_test)
    print(y_test)
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    
    print('predict_classes')
    y_pred = model.predict(x_test)
    predictions = np.round(y_pred)
    print(predictions)
    
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='binary')
    print(f'Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1_score:.2f}')
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    
    
    test_accuracyArr.append(test_accuracy * 100)
    test_f1_scoreArr.append(f1_score * 100)
    test_recallArr.append(recall * 100)
    test_precisionArr.append(precision * 100)
    i += 1
    
print("accuracyArr:   ", test_accuracyArr)
print("f1_scoreArr:   ", test_f1_scoreArr)
print("recallArr:     ", test_recallArr)
print("precisionArr:  ", test_precisionArr)
print(f'accuracy {AVG} lan: {sum(test_accuracyArr) / AVG}')
print(f'f1_score {AVG} lan: {sum(test_f1_scoreArr) / AVG}')
print(f'recall {AVG} lan: {sum(test_recallArr) / AVG}')
print(f'precision {AVG} lan: {sum(test_precisionArr) / AVG}')
print(f'total time {AVG} lan: {sum(TOTAL_TIME)}')

print('===========================================================================================================')