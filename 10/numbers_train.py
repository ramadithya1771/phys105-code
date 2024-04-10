from tensorflow import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

import matplotlib.pyplot as plt
import random as rnd

image_index = rnd.randrange(len(y_train))
print(y_train[image_index])
plt.imshow(x_train[image_index])
plt.show()

import numpy as np

print(x_train.shape)
print(x_test.shape)
print(y_train)
print(np.min(x_train[image_index]),np.max(x_train[image_index]))

# save input image dimensions
img_rows, img_cols = (28, 28)

x_train = x_train.reshape(x_train.shape[0], img_rows*img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows*img_cols)

x_train_arr = x_train.astype('float')/255.
x_test_arr = x_test.astype('float')/255.

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

num_classes = 10

model = Sequential()
model.add(Dense(15, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
epochs = 40

#model.fit(x_train_arr, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test_arr, y_test))
model.fit(x_train_arr, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
loss, accuracy = model.evaluate(x_test_arr, y_test)
print('Test loss:', loss)
print('Test accuracy: %.2f' % (accuracy*100))
model.save("test_model.h5")
