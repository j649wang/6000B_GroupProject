# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:51:14 2017

@author: wangjingyi
@reference code: https://github.com/pumpikano/tf-dann
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from flip_gradient import flip_gradient
'''
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import io
vgg = VGG16(weights='imagenet', include_top=False)
#graph = tf.Graph()
#sess  = tf.Session(graph = graph)
def create_dataset(file, is_test = False, source = False):
    A_train_f = open(file)
    A_train = A_train_f.readlines()
    A_train_f.close()
    A_train_name = list()
    A_train_label = np.array([])
    for line in A_train:
        A_train_name.append(line.split()[0])
        if(not is_test): 
            A_train_label = np.append(A_train_label, line.split()[1])

    size = len(A_train_name)
    if source: dirt = "Dataset_A\\dataresize_A"
    else: dirt = "Dataset_B\\dataresize_B" 
    for name,ii in zip(A_train_name, range(size)):   
        for filename in os.listdir(dirt):
            root, ext = os.path.splitext(filename)
            if root.startswith(name) and ext == '.png': break
        img_path = ''.join([dirt, '\\', filename])
        pic = feature_extract(img_path)
        if not ii: 
            train_img = pic
        else: 
            train_img = np.concatenate([train_img, pic])
    if (not is_test):
        labels  = np.zeros((size, 2))
        labels[np.arange(size), A_train_label.astype(int)] = 1
        return train_img, labels.astype(int)
    else:   
        return train_img
def feature_extract(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = vgg.predict(x)
    return features

#%%
# read datasetA
A_train, A_train_labels = create_dataset("Dataset_A\\train.txt", source = True)
A_valid, A_valid_labels = create_dataset("Dataset_A\\val.txt", source = True)
A_test = create_dataset("Dataset_A\\test.txt", is_test = True, source = True)
# read datasetB
B_train, B_train_labels = create_dataset("Dataset_B\\train.txt")
B_valid, B_valid_labels = create_dataset("Dataset_B\\val.txt")
B_test = create_dataset("Dataset_B\\test.txt", is_test = True)
# save file
np.save("A_train.npy", A_train)
np.save("A_train_labels.npy", A_train_labels)
np.save("A_valid.npy", A_valid)
np.save("A_valid_labels.npy", A_valid_labels)
np.save("A_test.npy", A_test)
np.save("B_train.npy", B_train)
np.save("B_train_labels.npy", B_train_labels)
np.save("B_valid.npy", B_valid)
np.save("B_valid_labels.npy", B_valid_labels)
np.save("B_test.npy", B_test)


# Compute pixel mean for normalizing data
pixel_mean = np.vstack([A_train, B_train]).mean()
'''
#%%
A_train= np.load("A_train.npy")
A_train_labels=np.load("A_train_labels.npy")
A_valid = np.load("A_valid.npy")
A_valid_labels = np.load("A_valid_labels.npy")
A_test = np.load("A_test.npy")
B_train = np.load("B_train.npy")
B_train_labels= np.load("B_train_labels.npy")
B_valid = np.load("B_valid.npy")
B_valid_labels = np.load("B_valid_labels.npy")
B_test = np.load("B_test.npy")
#%%
# Create a mixed dataset for TSNE visualization
num_test = 40
combined_test_imgs = np.vstack([A_valid[:num_test],B_valid[:num_test]])
combined_test_labels = np.vstack([A_valid_labels[:num_test], B_valid_labels[:num_test]])
combined_test_domain = np.vstack([np.tile([1., 0.], [num_test, 1]),
        np.tile([0., 1.], [num_test, 1])])

#%%
def batch_generator(data, batch_size, shuffle=True):
    ##Generate batches of data
    if shuffle:
        data = shuffle_aligned_list(data)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size >= len(data[0]):
            batch_count = 0

            if shuffle:
                data = shuffle_aligned_list(data)

        start = batch_count * batch_size
        end = start + batch_size
        batch_count += 1
        yield [d[start:end] for d in data]

def shuffle_aligned_list(data):
    ##Shuffle arrays in a list
    num = data[0].shape[0]
    p = np.random.permutation(num)
    return [d[p] for d in data]
#%%
batch_size = 32

class BreastModel(object):
    def __init__(self):
        self._build_model()
    
    def _build_model(self):
        self.X = tf.placeholder(tf.float64, [None, 7, 7, 512])
        self.y = tf.placeholder(tf.int64, [None, 2])
        self.domain = tf.placeholder(tf.float32, [None, 2])
        self.l = tf.placeholder(tf.float32, [])
        self.train = tf.placeholder(tf.bool, [])      
        X_input = tf.cast(self.X, tf.float32)# - pixel_mean
        self.y  = tf.cast(self.y, tf.float32)
        
        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor'):
            # Extract feature by vgg model
            # The domain-invariant feature
            self.feature = tf.reshape(X_input, [-1, 7*7*512])
            
        # MLP for class prediction
        with tf.variable_scope('label_predictor'):
            all_features = lambda: self.feature
            source_features = lambda: tf.slice(self.feature, [0, 0], [int(batch_size / 2), -1])
            classify_feats = tf.cond(self.train, source_features, all_features)
            
            all_labels = lambda: self.y
            source_labels = lambda: tf.slice(self.y, [0, 0], [int(batch_size / 2), -1])
            self.classify_labels = tf.cond(self.train, source_labels, all_labels)
            l_network = fully_connected(classify_feats, 512, activation='relu')
            logits = fully_connected(l_network, 2, activation='linear')            
            self.pred = tf.nn.softmax(logits)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits (logits=logits, labels = self.classify_labels)

        # MLP for domain prediction with adversarial loss
        with tf.variable_scope('domain_predictor'):
            # Flip the gradient when backpropagating through this operation
            feat = flip_gradient(self.feature, self.l)
            d_network = fully_connected(feat, 512, activation='relu')
            #d_network = fully_connected(d_network, 1024, activation='relu')
            d_logits = fully_connected(d_network, 2, activation='linear')          
            self.domain_pred = tf.nn.softmax(d_logits)
            self.domain_loss = tf.nn.softmax_cross_entropy_with_logits(logits=d_logits,labels = self.domain)
#%%
# Build the model graph
#tf.reset_default_graph()
graph = tf.get_default_graph()
with graph.as_default():
    model =  BreastModel()
    learning_rate = tf.placeholder(tf.float32, [])
    pred_loss = tf.reduce_mean(model.pred_loss)
    domain_loss = tf.reduce_mean(model.domain_loss)
    total_loss = pred_loss + domain_loss
    regular_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    dann_train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(pred_loss)
    # Evaluation
    correct_label_pred = tf.equal(tf.argmax(model.classify_labels, 1), tf.argmax(model.pred, 1))
    label_acc = tf.reduce_mean(tf.cast(correct_label_pred, tf.float32))
    correct_domain_pred = tf.equal(tf.argmax(model.domain, 1), tf.argmax(model.domain_pred, 1))
    domain_acc = tf.reduce_mean(tf.cast(correct_domain_pred, tf.float32))
#%%
# Params
num_steps = 1500

def train_and_evaluate(training_mode, graph, model, verbose=True):
    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run()

        # Batch generators
        gen_source_batch = batch_generator(
            [A_train, A_train_labels], int(batch_size / 2))
        gen_target_batch = batch_generator(
            [B_train, B_train_labels], int(batch_size / 2))
        gen_source_only_batch = batch_generator(
            [A_train, A_train_labels], batch_size)
        gen_target_only_batch = batch_generator(
            [B_train, B_train_labels], batch_size)

        domain_labels = np.vstack([np.tile([1., 0.], [int(batch_size / 2), 1]),
                                   np.tile([0., 1.], [int(batch_size / 2), 1])])

        # Training loop
        for i in range(num_steps):
            
            # Adaptation param and learning rate schedule as described in the paper
            p = float(i) / num_steps
            l = 2. / (1. + np.exp(-10. * p)) - 1
            lr = 0.01/ (1. + 10 * p)**0.75

            # Training step
            if training_mode == 'dann':

                X0, y0 = gen_source_batch.__next__()
                X1, y1 = gen_target_batch.__next__()
                X = np.vstack([X0, X1])
                y = np.vstack([y0, y1])

                _, batch_loss, dloss, ploss, d_acc, p_acc = \
                    sess.run([dann_train_op, total_loss, domain_loss, pred_loss, domain_acc, label_acc],
                             feed_dict={model.X: X, model.y: y, model.domain: domain_labels,
                                        model.train: True, model.l: l, learning_rate: lr})

                if verbose and i % 100 == 0:
                    print ('loss: %f  d_acc: %f  p_acc: %f  p: %f  l: %f  lr: %f' % \
                            (batch_loss, d_acc, p_acc, p, l, lr))

            elif training_mode == 'source':
                X, y = gen_source_only_batch.__next__()
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l, learning_rate: lr})

            elif training_mode == 'target':
                X, y = gen_target_only_batch.__next__()
                _, batch_loss = sess.run([regular_train_op, pred_loss],
                                     feed_dict={model.X: X, model.y: y, model.train: False,
                                                model.l: l, learning_rate: lr})

        # Compute final evaluation on test data
        source_acc = sess.run(label_acc,
                            feed_dict={model.X: A_valid, model.y: A_valid_labels,
                                       model.train: False})

        target_acc = sess.run(label_acc,
                            feed_dict={model.X: B_valid, model.y: B_valid_labels,
                                       model.train: False})
        
        test_domain_acc = sess.run(domain_acc,
                            feed_dict={model.X: combined_test_imgs,
                                       model.domain: combined_test_domain, model.l: 1.0})
        
        test_emb = sess.run(model.feature, feed_dict={model.X: combined_test_imgs})
        
    return source_acc, target_acc, test_domain_acc, test_emb



source_acc, target_acc, _, source_only_emb = train_and_evaluate('source', graph, model)
print('\nSource only traning:')
print('Source (A) accuracy:', source_acc)
print('Target (B) accuracy:', target_acc)


source_acc, target_acc, d_acc, dann_emb = train_and_evaluate('dann', graph, model)
print('\nDomain Adaptation training:')
print('Source (A) accuracy:', source_acc)
print('Target (B) accuracy:', target_acc)
print('Domain accuracy:', d_acc)
