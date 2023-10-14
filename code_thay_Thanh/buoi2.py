import pandas as pd
dulieu = pd.read_csv("winequality-red.csv", delimiter =";")



print ("so luong phan tu: ",len(dulieu))
h = dulieu.iloc[0]
print("so luong nhan: ",len(h))
import numpy as np
np.unique(dulieu)
dulieu.value_counts()

x = dulieu.iloc[:,:-1]
y = dulieu.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=4/10.0,random_state=5)
print("số lượng phần tử trong tập test ")
print(len(X_test))


from sklearn.neighbors import  KNeighborsClassifier
Mohinh_KNN =  KNeighborsClassifier(n_neighbors = 7)
Mohinh_KNN.fit(X_train,y_train)
# y_pred la ket qua du doan
y_pred = Mohinh_KNN.predict(X_test)

from sklearn.metrics import accuracy_score
print("cau d1: do chinh xac tong the:", accuracy_score(y_test,y_pred)*100)
from sklearn.metrics import confusion_matrix
cnf =confusion_matrix(y_test,y_pred)

normalized_confusion_matrix = cnf/cnf.sum(axis = 1, keepdims = True)
print('\ncau d1: độ chính xác của từng lớp:')
print(normalized_confusion_matrix)




from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

model = GaussianNB()
model.fit(X_train,y_train)

thucte = y_test
dubao = model.predict(X_test)
model.predict_proba(X_test)
print("cau e: do chinh xac tong the: ", accuracy_score(thucte,dubao)*100)
cnf =confusion_matrix(thucte,dubao)
normalized_confusion_matrix = cnf/cnf.sum(axis = 1, keepdims = True)
print('\ncau e: độ chính xác của từng lớp:')
print(normalized_confusion_matrix)

# cau f
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=1/3.0,random_state=5)
Mohinh_KNN =  KNeighborsClassifier(n_neighbors = 7)
Mohinh_KNN.fit(X_train,y_train)
# y_pred la ket qua du doan
y_pred = Mohinh_KNN.predict(X_test)
print("cau f: do chinh xac tong the KNN:", accuracy_score(y_test,y_pred)*100)

model = GaussianNB()
model.fit(X_train,y_train)

thucte = y_test
dubao = model.predict(X_test)
model.predict_proba(X_test)
print("cau f: do chinh xac tong the Bayes thơ ngây: ", accuracy_score(thucte,dubao)*100)
