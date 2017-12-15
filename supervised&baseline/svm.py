#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 13:35:21 2017

The base model using SVM and Bayes model, where Bayes is the metric of selecting
the model.

@author: yilu
"""

import cv2
import numpy as np
import os
import codecs
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB

# Define the file to load data
piclist = os.listdir("/Users/yilu/Downloads/dataresize_A")

train = codecs.open('train_data2.txt','r','utf-8')
test1 = codecs.open('val_dataB.txt','r','utf-8')
test2 = codecs.open('train_dataB.txt','r','utf-8')

path_list = []
label_list = []
training_set = []
test_list = []
test_data = []
test_label = []


# load the data and labels
for line in train.readlines():
    label = line.strip().split()
    path_list.append(label[0].decode('utf-8'))
    label_list.append(label[1].decode('utf-8'))

for line in test1.readlines():
    label = line.strip().split()
    test_list.append(label[0].decode('utf-8'))
    test_label.append(label[1].decode('utf-8'))

#for line in test2.readlines():
#    label = line.strip().split()
#    test_list.append(label[0].decode('utf-8'))
#    test_label.append(label[1].decode('utf-8'))
    
# build the training data
for i in range(len(path_list)):
    img = cv2.imread(path_list[i])
    img = cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xarr = np.squeeze(np.array(img).astype(np.float32))
    m,v = cv2.PCACompute(xarr, mean=None)
    arr = np.array(v)
    flat_arr = arr.ravel()
    training_set.append(flat_arr)

# build the test data
for i in range(len(test_list)):
    img = cv2.imread(test_list[i])
    img = cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    xarr = np.squeeze(np.array(img).astype(np.float32))
    m,v = cv2.PCACompute(xarr, mean=None)
    arr = np.array(v)
    flat_arr = arr.ravel()
    test_data.append(flat_arr)

# transform the data into correct format
features = np.array(training_set)
trainLabel = np.array([np.float32(label_list)]).reshape((-1,1))
labels = np.ravel(trainLabel)
testlabels = np.array([np.float32(test_label)]).reshape((-1,1))
testlabels = np.ravel(testlabels)
testdata = np.array(test_data)

# using PCA to reduce the dimension of the image data
pca = PCA(n_components=100)
pca.fit(features)
features = pca.transform(features)
testData = pca.transform(testdata)

scaler = StandardScaler().fit(features)
features = scaler.transform(features)
testData = scaler.transform(testData)

# Initialization
svm_acc_trn = np.array([])
svm_acc_tst = np.array([])

bayes_acc_trn = np.array([])
bayes_acc_tst = np.array([])

# Cross-validation
kfold = KFold(n_splits=10, shuffle=True)

predict = []
predict_bayes = []
cv = 0
for train, test in kfold.split(features):
    train_features = features[train, :]
    train_target = labels[train]
    test_features = features[test, :]
    test_target = labels[test]
        
    # SVM model
    svm = SVC(kernel='rbf', C=0.5, random_state=1)
    svm.fit(train_features, train_target)
    
    # Bayes model
    bnb = BernoulliNB()
    bnb.fit(train_features, train_target)
    
    train_acc_svm = np.mean(svm.predict(train_features) == train_target)
    test_acc_svm = np.mean(svm.predict(test_features) == test_target)
    svm_acc_trn = np.append([svm_acc_trn], [train_acc_svm])
    svm_acc_tst = np.append([svm_acc_tst], [test_acc_svm])   
    
    train_acc_bayes = np.mean(bnb.predict(train_features) == train_target)
    test_acc_bayes = np.mean(bnb.predict(test_features) == test_target)
    bayes_acc_trn = np.append([bayes_acc_trn], [train_acc_bayes])
    bayes_acc_tst = np.append([bayes_acc_tst], [test_acc_bayes])
 
    print("%d Fold svm Train accuracy:%f, Test accuracy:%f" % (cv, train_acc_svm, test_acc_svm))
    print("%d Fold bayes Train accuracy:%f, Test accuracy:%f" % (cv, train_acc_bayes, test_acc_bayes))
    cv+=1

print("svm train accuracy: %f, Test accuracy: %f" % (np.mean(svm_acc_trn), np.mean(svm_acc_tst)))
print("bayes train accuracy: %f, Test accuracy: %f" % (np.mean(bayes_acc_trn), np.mean(bayes_acc_tst)))

# Train a svm model with total data and predict the validation data from
# dataset B
svm2 = SVC(kernel='rbf', C=0.5, random_state=0)
svm2.fit(features, labels)
test_results = svm2.predict(testData)
acc = np.mean(test_results == testlabels)
print("svm predict accuray: %f" % acc)

# Train a Bayes model with total data and predict the validation data from
# dataset_B
bnb2 = BernoulliNB()
bnb2.fit(features, labels)
test_results2 = bnb2.predict(testData)
acc2 = np.mean(test_results2 == testlabels)
print("Bayes predict accuracy: %f" % acc)
