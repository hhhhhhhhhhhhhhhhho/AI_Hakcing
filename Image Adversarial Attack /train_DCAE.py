import keras
import tensorflow as tf
import os
import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.image as mpimg
from sklearn import preprocessing
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D,Conv3D, MaxPooling2D, Dropout, Flatten, Conv2DTranspose,GaussianNoise
from keras.callbacks import ModelCheckpoint, EarlyStopping

import matplotlib.pyplot as plt
import os
import numpy
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

from keras.callbacks import TensorBoard

### autonomical image resize

from cv2 import cv2
import os
def resizeImg():
  # we'll fill in this section below
  return

def resizeImg(image, width=None, height=None):
  dim=None
  (h,w) = image.shape[:2]
  if width is None and height is None:
    return image
  
  elif width is not None and height is not None:
    dim = (width, height)
  
  elif width is None:
    r = height/float(h)
    dim = (int(w*r), height)
  else:
    r = width / float(w)
    dim = (width, int(h*r))

  resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
  return resized

def main():

  folder = "./originalImgs/"
  newResizedFolder = "./newResizedImgs/"
  for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, filename))
    if img is not None:
      newImage = resizeImg('/content/Unknown-6.png')
      newImgPath = newResizedFolder + filename
      cv2.imwrite(newImgPath, newImage)
      

import numpy as np
def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 64, 64, 3))
    return array

def solo_preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = np.reshape(array, (1,64, 64, 3))
    return array

def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


X_train = train_images 
X_test = test_images

noised_X_train = GaussianNoise(0.2)(X_train, training=True)
noised_X_test = GaussianNoise(0.2)(X_test, training=True)

from keras import backend as K
K.clear_session()

X_train = preprocess(X_train)
#nosied_X_train = preprocess(noised_X_train)
X_test = preprocess(X_test)
#noised_X_test = preprocess(X_test)

#FIXME
#Gaussian noise 넣을 때 전처리 함. 

# Encoder 
with tf.device('/gpu:0'): 
  input = layers.Input(shape=(64, 64, 3))

  # Encoder
  x = layers.Conv2D(32, (3, 3), activation="selu", padding="same")(input)
  x = layers.MaxPooling2D((2, 2), padding="same")(x)
  x = layers.Conv2D(32, (3, 3), activation="selu", padding="same")(x)
  x = layers.MaxPooling2D((2, 2), padding="same")(x)


  # Decoder
  x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="selu", padding="same")(x)
  x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="selu", padding="same")(x)
  x = layers.Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

  # Autoencoder
  autoencoder = Model(input, x)
  autoencoder.compile(optimizer="adam", loss="mse")
  autoencoder.summary()


history=autoencoder.fit(
    x=noised_X_train,
    y=X_train,
    epochs=10,
    batch_size=64,
    shuffle=True,
    validation_data=(noised_X_test, X_test)
)
