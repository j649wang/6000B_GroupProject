#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 10:18:08 2017

This file is used to extract the features of images by using VGG16 pre-trained
model. 

@author: yilu
"""

import os
import codecs
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16

# load the path of the pictures
piclist = os.listdir("/Users/yilu/Downloads/dataresize_A")

labels1 = codecs.open('train_data.txt','r','utf-8')
path_list = []

# store the paths into list
for line in labels1.readlines():
    label = line.strip().split()
    path_list.append(label[0].encode('utf-8'))

# load the pre-trained VGG16 model 
model = VGG16(weights='imagenet', include_top=False)

# read the images and using the VGG16 to extract the featuers 
img_list = []
for i in range(len(path_list)):
    img = image.load_img(path_list[i], target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    img_list.append(features)
    
# Store the features as array
np.save('train.npy', np.array(img_list).reshape(410,7,7,512))