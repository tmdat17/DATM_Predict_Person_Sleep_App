import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
# Danh gia Precision, Recall, and Threashold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import time
import joblib

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t

    avg = sum_num / len(num)
    return avg
n = 0
KetquaNlan = []
KetquaNlanF1 = []
KetquaNlanRecall = []
KetquaNlanPrecision = []
TOTAL_TIME = []
K = 20
N_NEIGHBORS = 20
N_AVERAGE = 50
MAX_ACC = -999999
while (n < N_AVERAGE):
    n += 1
    print(f'LAN {n} --------------------------------------------------------------------------------------')
    # -----------------------------
    # Doc du lieu tu file
    dulieuload = pd.read_csv("../data_train_text/merge_2_lie_to_train.txt", delimiter=' ')
    # dulieuload.replace({'s': 1, 'w': 0}, inplace=True)
    # print(dulieuload)
    dulieu_x = dulieuload.iloc[:, 0:-1]
    # Doc cot label 
    dulieu_y = dulieuload.iloc[:, -1]
    # print('label:  ', dulieu_y)

    print('len full data: ', len(dulieuload))
    print('len x data: ', len(dulieu_x))
    print('len y data: ', len(dulieu_y))
    
    # Gia tri cua cac nhan unique
    DanhSachY_PhanBiet = np.unique(dulieu_y)

    # ----------------------------
    # Chia tap du lieu
    # Danh gia nghi thuc K-fold
    kf = KFold(n_splits=K, shuffle=True)
    for idtrain, idtest in kf.split(dulieuload):
        x_train = dulieu_x.iloc[idtrain,]
        x_test = dulieu_x.iloc[idtest,]
        y_train = dulieu_y.iloc[idtrain]
        y_test = dulieu_y.iloc[idtest]
    
    print(f'Kfold K = {K}')
    print('len x_train:  ', len(x_train))
    print('len y_train:  ', len(y_train))
    print('len x_test:  ', len(x_test))
    print('len y_test:  ', len(y_test))
    # print("tap du lieu x test \n",x_test)
    # print("tap du lieu y test label \n",y_test)

    # -------------------------------
    # Load mo hinh KNN
    each_time_cur = time.time()
    Model_Sleep = KNeighborsClassifier(n_neighbors = N_NEIGHBORS)
    Model_Sleep.fit(x_train, y_train)
    end_each_time = time.time() - each_time_cur
    print(f'time train lan {n}:  ', end_each_time)
    TOTAL_TIME.append(end_each_time)

    # --------------------------------
    # Du doan mo hinh 
    print('x_test:  \n', x_test)
    Y_dudoan = Model_Sleep.predict(x_test)
    print("Ket qua du doan", Y_dudoan)
    # ----------------------------------
    precision, recall, f1, support = precision_recall_fscore_support(y_test, Y_dudoan)
    accuracy = accuracy_score(y_test, Y_dudoan)*100
    
    if(accuracy >= 95 and accuracy > MAX_ACC):
        joblib.dump(Model_Sleep, f'../model_detect_sleep/KNN/KNN_KFold_K_{K}_N_NEIGHBORS_{N_NEIGHBORS}.h5')
        MAX_ACC = accuracy
    # In kết quả
    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-score:", f1)
    # print("Accuracy:", accuracy)
    
    KetquaNlan.append(accuracy)
    KetquaNlanF1.append(f1)
    KetquaNlanRecall.append(recall)
    KetquaNlanPrecision.append(precision)

print (f"Danh gia mo hinh KNN\nk = {K}\nN_AVERAGE = {N_AVERAGE}\nN_NEIGHBORS = {N_NEIGHBORS}")
# print("Ket qua N lan accuracy", KetquaNlan)
# print("Ket qua N lan F1", KetquaNlanF1)
# print("Ket qua N lan Recall", KetquaNlanRecall)
# print("Ket qua N lan Precision", KetquaNlanPrecision)
print(f'-------------------------------------- Ket Qua {N_AVERAGE} lan ------------------------------')
print(f'{N_AVERAGE} lan accuracy: {KetquaNlan}')
print(f'{N_AVERAGE} lan F1: {KetquaNlanF1}')
print(f'{N_AVERAGE} lan Recall: {KetquaNlanRecall}')
print(f'{N_AVERAGE} lan Precision: {KetquaNlanPrecision}')

print('-------------------------------------- Trung Binh ------------------------------')
print(f"trung binh {N_AVERAGE} lan accuracy", cal_average(KetquaNlan))
print(f"trung binh {N_AVERAGE} lan F1", cal_average(cal_average(KetquaNlanF1)))
print(f"trung binh {N_AVERAGE} lan Recall", cal_average(cal_average(KetquaNlanRecall)))
print(f"trung binh {N_AVERAGE} lan Precision", cal_average(cal_average(KetquaNlanPrecision)))
print(f"MAX ACCURACY IN {N_AVERAGE}:  {MAX_ACC}")
print(f'TOTAL TIME {N_AVERAGE} lan:  ', sum(TOTAL_TIME))
# ------------------------------------
# Ma tran Nham lan (Loi), Confusion matrix
# ThucTe = y_test
# DuBaoKetQua = Model_Sleep.predict(x_test)
# MaTranKetQua = confusion_matrix(ThucTe,DuBaoKetQua)
# print(MaTranKetQua)

