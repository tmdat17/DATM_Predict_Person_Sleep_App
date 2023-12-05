import numpy as np
import pandas as pd

# Load du lieu va chia du lieu
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# KNN
from sklearn.neighbors import KNeighborsClassifier

# Danh gia mo hinh - Do chinh xac
from sklearn.metrics import accuracy_score

# confusion Matrix
from sklearn.metrics import confusion_matrix

# Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

# DecisionTree
from sklearn.tree import DecisionTreeClassifier

# Danh gia nghi thuc K-fold
from sklearn.model_selection import KFold

# Danh gia Precision, Recall, and Threashold
from sklearn.metrics import precision_recall_fscore_support

def cal_average(num):
    sum_num = 0
    for t in num:
        sum_num = sum_num + t           

    avg = sum_num / len(num)
    return avg

n = 0
Ketqua50lan = []
while (n < 50):
    n += 1
    # -----------------------------
    # Doc du lieu tu file
    dulieuload = pd.read_csv("winequality-white.csv", delimiter=';')
    # print(dulieuload)
    dulieu_x = dulieuload.iloc[:, 0:-1]
    # Doc cot cuoi cung
    dulieu_y = dulieuload.iloc[:, -1]
    # -------------------------------

    # print("so luong thuoc tinh", len(dulieuload.iloc[0]))

    # Gia tri cua cac nhan
    DanhSachY_PhanBiet = np.unique(dulieu_y)
    # print("Gia tri cua cac nhan", DanhSachY_PhanBiet)

    # print("So luong cua moi lop:",dulieu_y.value_counts())

    # ----------------------------
    # Chia tap du lieu
    # Danh gia nghi thuc K-fold
    kf = KFold(n_splits=60, shuffle=True)
    for idtrain, idtest in kf.split(dulieuload):
        x_train = dulieu_x.iloc[idtrain,]
        x_test = dulieu_x.iloc[idtest,]
        y_train = dulieu_y.iloc[idtrain]
        y_test = dulieu_y.iloc[idtest]
    # print("tap du lieu trong tap train \n",x_train)
    # print("tap du lieu trong tap test \n",x_test)

    # -------------------------------
    # Load mo hinh
    MoHinhDT = DecisionTreeClassifier(
        criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
    MoHinhDT.fit(x_train, y_train)
    # --------------------------------
    # Du doan mo hinh
    Y_dudoan = MoHinhDT.predict(x_test)
    # print("Ket qua du doan", Y_dudoan)
    # ----------------------------------
    KetQua_DoChinhXac = accuracy_score(y_test, Y_dudoan)*100
    # print("Ket qua do chinh Xac", KetQua_DoChinhXac)
    Ketqua50lan.append(KetQua_DoChinhXac)

print("Ket qua 50 lan", Ketqua50lan)
print("Ket qua do chinh xac trung binh 50 lan", cal_average(Ketqua50lan))














# ------------------------------------
# Ma tran Nham lan (Loi), Confusion matrix
# ThucTe = y_test
# DuBaoKetQua = MoHinhDT.predict(x_test)
# MaTranKetQua = confusion_matrix(ThucTe,DuBaoKetQua)
# print(MaTranKetQua)


# ---------------------------------------
# Danh gia cho 8 phan tu dau tien trong tap test
# x_test = x_test[0:8]
# y_test = y_test[0:8]
# print("8 phan tu dau tien trong tap test",x_test)
# Y_dudoan = MoHinhDT.predict(x_test)
# print("ket qua du doan",Y_dudoan)


# -----------------------------
# prec, rec, fsco, sup = precision_recall_fscore_support(y_test,Y_dudoan)
# print("precison", prec)
