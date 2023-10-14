import tensorflow as tf
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split


class MobileNetV1_1D(tf.keras.Model):
    def __init__(self, num_classes):
        super(MobileNetV1_1D, self).__init__()

        # Define the layers of the model
    
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(20, 1)),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=2, activation='relu')
           ])
        self.bottleneck1 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu'),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, strides=1, activation='relu')
        ])
        self.bottleneck2 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=2, activation='relu'),
            tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu')
        ])
        self.bottleneck3 = tf.keras.Sequential([
            tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=2, activation='relu'),
            tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu')
        ])
        self.avgpool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bottleneck1(x)
        # x = self.bottleneck2(x)
        # x = self.bottleneck3(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x


dulieuload = pd.read_csv("../data_train_text/merge_2_lie_to_train.txt", delimiter=' ')
dulieuload.replace({'s': 1, 'w': 0}, inplace=True)
X_train,X_test,y_train,y_test =train_test_split(dulieuload.iloc[:, 0:-1],dulieuload.iloc[:,-1],test_size = 1/3.0,random_state = 2, shuffle=True)
print(set(dulieuload.iloc[:,-1]))

model = MobileNetV1_1D(num_classes=1)

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(X_test, y_test)

# Make predictions
predictions = model.predict(X_test)