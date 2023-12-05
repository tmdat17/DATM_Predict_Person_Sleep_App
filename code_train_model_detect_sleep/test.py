import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import Sequential, Model
from sklearn.svm import SVC,SVR

# Generate synthetic data for binary classification
dulieuload = pd.read_csv("../data_train_text/merge_2_lie_to_train.txt", delimiter=' ')
dulieuload.replace({'s': 1, 'w': 0}, inplace=True)
X_train,X_test,y_train,y_test =train_test_split(dulieuload.iloc[:, 0:-1],dulieuload.iloc[:,-1],test_size = 0.2,random_state = 2, shuffle=True)
print(set(dulieuload.iloc[:,-1]))

np.random.seed(0)
# X_train = np.random.rand(666, 20)  # 1000 samples with 100 features
# y_train = np.random.randint(2, size=666)  # Binary labels (0 or 1)
print('x:  ', X_train)
print('y:  ', y_train)
# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the 1D CNN model
model = keras.Sequential([
    keras.layers.Input(shape=(20, 1)),  # Input shape (100 features, 1 channel)
    keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Reshape the data to match the input shape (batch_size, sequence_length, input_dimension)
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2,  verbose=1, callbacks=[early_stopping])

# print("bat dau trich dac trung: ")
# new_model=Model(inputs=model.input,outputs=model.get_layer('dense').output)

# # print("new_model:  ", new_model)
# # new_model.save('./model_extract_feature_CNN/lenet_extract_features_epo{}_bs{}.h5'.format(EPOCHS, BS))
# feat_train=new_model.predict(X_train)
# print("feat_train",feat_train.shape)

# feat_test=new_model.predict(X_test)
# print("feat_test",feat_test.shape)


# model_SVM=SVC(kernel="rbf", C=10000, gamma=0.01)
# model_SVM.fit(feat_train, y_train)
# prepY=model_SVM.predict(feat_test)



# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Make predictions
predictions = model.predict(X_test)