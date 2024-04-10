from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

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

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

x_train_arr = x_train.astype('float')/255.
x_test_arr = x_test.astype('float')/255.

num_classes = 10

y_train_arr = keras.utils.to_categorical(y_train, num_classes)
y_test_arr = keras.utils.to_categorical(y_test, num_classes)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_rows, img_cols, 1)))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(64, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Flatten())
#model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 128
epochs = 5

model.fit(x_train_arr, y_train_arr, batch_size=batch_size, epochs=epochs, validation_split=0.1)
#model.fit(x_train_arr, y_train_arr, batch_size=batch_size, epochs=epochs, validation_data=(x_test_arr, y_test_arr))
loss, accuracy = model.evaluate(x_test_arr, y_test_arr)
print('Test loss:', loss)
print('Test accuracy: %.2f' % (accuracy*100))
model.save("test_model_conv.h5")
