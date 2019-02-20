### kaggle digit recognizer


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

X_train = (dataset_train.iloc[:, 1:].values).astype('float32')
y_train = dataset_train.iloc[:, 0].values.astype('int32')
X_test = dataset_test.values.astype('float32')

X_train = X_train.reshape(X_train.shape[0], 28, 28)
X_test = X_test.reshape(X_test.shape[0], 28, 28)

for i in range(6, 9):
    plt.subplot(330 + (i+1))
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
    plt.title(y_train[i])

plt.show()
# preprocessing

mean_px = X_train.mean().astype(np.float32)
std_px = X_train.std().astype(np.float32)

def standardize(x):
    return (x-mean_px)/std_px

# one hot encoding labels


from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train)
num_classes = y_train.shape[1]

plt.title(y_train[9])
plt.plot(y_train[9])
plt.xticks(range(10))
plt.show()

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


