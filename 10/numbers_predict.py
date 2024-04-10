import imageio
import numpy as np
from matplotlib import pyplot as plt

im = imageio.imread("a3Rql9C.png")

gray = im[...,0]
plt.imshow(gray, cmap = plt.get_cmap('gray'))
plt.show()

# reshape the image
img_rows, img_cols = (28, 28)
gray = gray.reshape(1, img_rows*img_cols)

# normalize image
gray_arr = gray.astype('float')/255.

# load the model
from tensorflow import keras
model = keras.models.load_model("test_model.h5")

# predict digit
prediction = model.predict(gray_arr)
print(prediction)
print(prediction.argmax())
