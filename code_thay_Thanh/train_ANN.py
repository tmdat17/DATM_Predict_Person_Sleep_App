import numpy as np
import pandas as pd

# Load du lieu va chia du lieu
from sklearn.model_selection import train_test_split
# KNN
from sklearn.neighbors import KNeighborsClassifier
import keras
# Danh gia mo hinh - Do chinh xac
from sklearn.metrics import accuracy_score

# confusion Matrix
from sklearn.metrics import confusion_matrix

# Bayes
# from sklearn.naive_bayes import GaussianNB
# from sklearn.naive_bayes import MultinomialNB

# DecisionTree
from sklearn.tree import DecisionTreeClassifier

# Danh gia nghi thuc K-fold
from sklearn.model_selection import KFold

# Danh gia Precision, Recall, and Threashold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils import shuffle

# Doc du lieu tu file
dulieuload = pd.read_csv("../data_train_text/merge_2_lie_to_train.txt", delimiter=' ')
dulieuload = dulieuload.iloc[:,:]
print("6  dong dau full\n", dulieuload)

# Sữ dụng nghi thức Hold-out phân chia tập dữ liệu huấn luyện
x_train,x_test,y_train,y_test =train_test_split(dulieuload.iloc[:, 0:-1],dulieuload.iloc[:,-1],test_size = 1/3.0,random_state = 2, shuffle=True)
x_train.iloc[1:6,]
print("6 dong dau x_train: \n", len(x_train))
print("6 dong dau x_test: \n", len(x_test))


from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# regressor = DecisionTreeRegressor(random_state = 0)
# regressor.fit(x_train, y_train)

# model = DecisionTreeClassifier(random_state=42)
# model = RandomForestClassifier(n_estimators=2, random_state=42)

# model.fit(x_train, y_train)

# # Dự đoán và đánh giá mô hình
# y_pred = model.predict(x_test)
# print(y_test[1:6])

# accuracy = accuracy_score(y_test, y_pred)
# print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

train_data, train_labels, validation_data, validation_labels, test_data, test_labels = train_test_split(padded_sequences, [], test_size=0.25)

# Define the CNN architecture
model = keras.Sequential([
    keras.layers.Conv1D(32, kernel_size=5, activation='relu', input_shape=(128, 128)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, ['s', 'w'], epochs=10)

# Evaluate the model on the validation set
# val_loss, val_accuracy = model.evaluate(validation_data, validation_labels)

# Test the model on the test set
test_loss, test_accuracy = model.evaluate(x_train, ['s', 'w'])

# Print the test accuracy
print('Test accuracy:', test_accuracy)

# Đánh giá kết quả dự đoán
# from sklearn.metrics import mean_squared_error
# err = mean_squared_error(y_test,y_pred)
# err
# print("ket qua du doan qua chỉ số MSE ", (err*100))

# print("ket qua du doan qua chỉ số RMSE ",(np.sqrt(err)*100))