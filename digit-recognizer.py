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

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

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

# preprocessing
seed = 43
np.random.seed(seed)

# import Linear model
from keras.models import Sequential
from keras.layers.core import Lambda, Dense, Flatten, Dropout
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization, Convolution2D, MaxPooling2D


model = Sequential()
model.add(Lambda(standardize, input_shape=(28, 28, 1)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

from keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing import image

gen = image.ImageDataGenerator()

# cross validation

from sklearn.model_selection import train_test_split

X = X_train
y = y_train

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

batches = gen.flow(X_train, y_train, batch_size=64)
val_batches = gen.flow(X_test, y_test, batch_size=64)

history = model.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=3,
                              validation_data=val_batches, validation_steps=val_batches.n)


history_dict = history.history


loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)


import matplotlib.pyplot as plt
plt.clf()
# "bo" is for "blue dot"
plt.plot(epochs, loss_values, 'bo')
# b+ is for "blue crosses"
plt.plot(epochs, val_loss_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.show()

plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc_values, 'bo')
plt.plot(epochs, val_acc_values, 'b+')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

predictions = model.predict_classes(X_test, verbose=0)
submission = pd.DataFrame({'ImageId': list(range(1, len(predictions) + 1)),
                           'Label': predictions})
submission.to_csv('DR.csv', index=False, header=True)