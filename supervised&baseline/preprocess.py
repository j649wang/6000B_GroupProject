#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 17:28:10 2017

This file is used to preprocess the data.

@author: yilu
"""
import codecs
import os

train = codecs.open('train.txt','r','utf-8')
val = codecs.open('val.txt', 'r', 'utf-8')
test = codecs.open('test.txt','r','utf-8')
piclist = os.listdir("./dataresize_Ayn")


match_list = []
label_list = []
val_list = []
val_label = []
test_list = []
test_label = []


# load the name of pictures as well as the labels
for line in train.readlines():
    label = line.strip().split()
    match_list.append(label[0].encode('utf-8'))
    label_list.append(label[1].encode('utf-8'))
    
for line in val.readlines():
    label = line.strip().split()
    val_list.append(label[0].encode('utf-8'))
    val_label.append(label[1].encode('utf-8'))
    
for line in test.readlines():
    label = line.strip().split()
    test_list.append(label[0].encode('utf-8'))
    

# combine the path of training data and its label
train_data = []
output_file = codecs.open("train_data.txt",'w','utf-8')
for i in range(len(piclist)):
    for j in range(len(match_list)):
        if piclist[i][:8] == match_list[j][:8]:
            train_data.append([piclist[i], label_list[j]])
            output_file.write('./dataresize_A/'+piclist[i]+'\t'+label_list[j]+'\n')


# combine the path of validation data and its label
output_file2 = codecs.open("val_data.txt",'w','utf-8')
valData = []
for i in range(len(piclist)):
    for j in range(len(val_list)):
        if piclist[i][:8] == val_list[j][:8]:
            valData.append([piclist[i], val_label[j]])
            output_file2.write('./dataresize_A/'+piclist[i]+'\t'+val_label[j]+'\n')
 
# combine the path of test data and its label        
testData = []
output_file3 = codecs.open("test_data.txt",'w','utf-8')
for i in range(len(test_list)):
    testData.append(test_list[i])
    output_file3.write('./dataresize_A/'+test_list[i]+'\n')
