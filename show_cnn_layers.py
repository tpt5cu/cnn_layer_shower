from keras.applications import vgg19
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions
import matplotlib.pyplot as plt
import numpy as np
import ssl
import os
from os.path import join as pJoin
import argparse
from keras.models import load_model
import random
from keras import models
from keras import layers
from keras import optimizers
from keras.models import Model



ssl._create_default_https_context = ssl._create_unverified_context


def import_image(imgpath):
	# load an image in PIL format
	original = load_img(imgpath, target_size=(224, 224))
	print('PIL image size',original.size)
	numpy_image = img_to_array(original)
	numpy_image = np.expand_dims(numpy_image, axis=0)
	return numpy_image

# #Load Model
model = vgg19.VGG19(weights='imagenet')

imgpath = '/Users/tuomastalvitie/Documents/Images_copy/Clothing/hawaiin.jpg'


# with a Sequential model
layer_name = 'pool5'
intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.get_layer('block2_pool').output)
intermediate_output = intermediate_layer_model.predict(import_image(imgpath))[0]

print intermediate_output.shape

r = intermediate_output.shape[2]
w=10
h=10
fig=plt.figure(figsize=(8,8))
columns=4
rows=5
for i in range(1, columns*rows +1):
	fig.add_subplot(rows, columns, i)
	j = random.randint(0,r)
	plt.imshow(np.uint8(intermediate_output[:,:,j]), cmap='jet')
plt.show()

