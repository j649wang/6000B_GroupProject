#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:47:58 2017

This file is to use fine tune to realize the supervised domain adaptation.

@author: yilu
"""

import tensorflow as tf
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

tf.reset_default_graph()

size = 7

# load the data
#inputs = np.load('A_train.npy')
inputs = np.load('trainB.npy')
labels = np.load('A_train_labels.npy')
val = np.load('A_valid.npy')
val_label = np.load('A_valid_labels.npy')

Binputs = np.load('B_train.npy')
Blabels = np.load('B_train_labels.npy')
Bval = np.load('B_valid.npy')
Bval_label = np.load('B_valid_labels.npy')

# Define the layers of CNN
cnn = input_data(shape=[None, size, size, 512])
        
cnn = fully_connected(cnn, 1024, activation='relu')
cnn = fully_connected(cnn, 512, activation='relu')#256
cnn = dropout(cnn, 1)

cnn = fully_connected(cnn, 2, activation='softmax')

cnn = regression(cnn, optimizer='adam',#adam
                 loss='categorical_crossentropy',
                 learning_rate=0.0001)#0.0001
 
# Train the last fc layers of CNN
model = tflearn.DNN(cnn, checkpoint_path='model_finetuning',
                    max_checkpoints=3, tensorboard_verbose=0)

# Fine tune the CNN based on source dataset
model.fit(inputs, labels, n_epoch=5, batch_size=50, shuffle=True, 
          validation_set=(val, val_label),
          show_metric=True, run_id='cnn')

# Fine tune the CNN based on the target dataset
model.fit(Binputs, Blabels, n_epoch=10, batch_size=50, shuffle=True,
          validation_set=(Bval, Bval_label),
            show_metric=True, run_id='cnn')

# using the model to predict labels
Btest = np.load('B_test.npy')
results = model.predict(Btest)