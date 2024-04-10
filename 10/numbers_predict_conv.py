import numpy as np
from matplotlib import pyplot as plt

#from tensorflow import keras
#
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#
#import random as rnd
#
#test_index = rnd.randrange(len(y_test))
#plt.imshow(x_test[test_index,...])
#print(y_test[test_index])
#plt.show()
#
## reshape the image
#img_rows, img_cols = (28, 28)
#test_im = np.expand_dims(np.expand_dims(x_test[test_index,...], -1),0)

import imageio
im = imageio.imread("a3Rql9C.png")
gray = im[...,0]
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

# reshape the image
img_rows, img_cols = (28, 28)
test_im = gray.reshape(1, img_rows, img_cols, 1)

# normalize image
test_im_arr = test_im.astype('float')/255.

# load the model
from tensorflow import keras
model = keras.models.load_model("test_model_conv.h5")

# predict digit
prediction = model.predict(test_im_arr)
print(prediction)
print(prediction.argmax())
