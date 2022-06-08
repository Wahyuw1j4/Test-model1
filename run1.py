from tkinter.tix import IMAGE
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


food_list = ['Ayam Betutu','Beberuk Terong','Coto Makassar','Gudeg','Kerak Telor','Mie Aceh','Nasi Kuning','Nasi Pecel','Papeda','Pempek','Peuyeum','Rawon','Rendang','Sate Madura','Serabi','Soto Banjar','Soto Lamongan','Tahu Sumedang']
model = load_model('model.h5',compile = False)
img = 'pic1.jpeg'
img = image.load_img(img, target_size=(224, 224))
img = image.img_to_array(img) / 255.0                  
img = tf.expand_dims(img, axis=0)                                              


pred = model.predict(img)
index = np.argmax(pred)

prediction = model(img)
pred_idx = np.argmax(prediction)
accuration = prediction[0][pred_idx] * 100

food_list.sort()
pred_value = food_list[index]

plt.imshow(img[0])                           
plt.axis('off')
plt.title(f'Prediction: {pred_value}, Acc: {accuration}')
plt.show()

