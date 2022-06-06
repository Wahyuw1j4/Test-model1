import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.applications.vgg16 import VGG16
import os

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# def decode_image(filename, image_size=(512, 512)):
#     bits = tf.io.read_file(filename)
#     image = tf.image.decode_image(bits, channels=3)
#     image = tf.cast(image, tf.float32) / 255.0
#     image = tf.expand_dims(image, axis=0) 
#     image = tf.image.resize(image, image_size)

#     return image


# img = decode_image('pic1.jpeg')
# # model = keras.models.load_model("model.h5")
# model = VGG16(weights = 'model.h5')
# print(model.predict(img, verbose=1))

#Function to load the model 
def load_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

print(load_model("bestmodel_18class.hdf5"))

