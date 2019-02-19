### kaggle digit recognizer







import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

X_train = []
y_train = []

N = 42000
D = 28000


for i in range(0, N):
    rdata = dataset_train.iloc[i, :].values
    y_train.append(rdata[0])
    X_row = []
    for n in range(0, 28):
        row_n = []
        for rw in range(0, 28):
            row_n.append(rdata[n*28 + rw])
        X_row.append(row_n)
    X_train.append(X_row)

X_test = []
y_test = []

for i in range(0, D):
    rdata = dataset_test.iloc[i, :].values
    y_test.append(rdata[0])
    X_row = []
    for n in range(0, 28):
        row_n = []
        for rw in range(0, 28):
            row_n.append(rdata[n*28 + rw])
        X_row.append(row_n)
    X_test.append(X_row)


X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# preprocessing

# cnn

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten

classifier = Sequential()

# 1 convolution
classifier.add(Convolution2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# 1 maxpool
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# flatten
classifier.add(Flatten())

# ann

# full connection

classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=9, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))


