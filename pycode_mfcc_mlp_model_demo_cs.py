#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 3 2019

techds: mfcc mlp model

"""



import numpy as np
np.random.seed(200) 
import tensorflow as tf
from tensorflow import set_random_seed 
set_random_seed(300)

from keras.utils import np_utils
from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, \
                         MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.models import load_model
from keras.models import Sequential

from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sh

import librosa
import librosa.display
import os
import pandas as pd

import re

import warnings
warnings.filterwarnings('ignore')

from pydub import AudioSegment as audio
from pydub.playback import play 


def tag_cols(df,beg_str,last_col_name):
    colnames = []
    beg_str = str(beg_str)
    len_col = len(df.columns)
    for l in range(len_col-1 ):
        name = beg_str + str(l+1)
        colnames.append(name)
    last_col_name = str(last_col_name)
    colnames.append(last_col_name)
    df.columns = colnames


# need 2 .npy files
# features_mfcc_train.npy
# features_mfcc_test.npy
    
# create df of training data features and labels from saved features file.
tr_mfcc = np.load('./Desktop/features_mfcc_train.npy')
tr_mfcc_df = pd.DataFrame(list(map(np.ravel, tr_mfcc)))
tr_mfcc_df.columns = ['feature','label']

# split feature col into individual cols and then combine with label col
tr_mfcc_df2 = pd.DataFrame(tr_mfcc_df['feature'].values.tolist() )
trainf_df = pd.concat([tr_mfcc_df2, tr_mfcc_df.label], axis=1)

# tag feature columns with f + num labels
tag_cols(trainf_df,'f','label') 

# create df of training data features and labels from saved features file.
te_mfcc = np.load('./Desktop/features_mfcc_test.npy')
te_mfcc_df = pd.DataFrame(list(map(np.ravel, te_mfcc)))
te_mfcc_df.columns = ['feature','label']

# split feature col into individual cols and then combine with label col
te_mfcc_df2 = pd.DataFrame(te_mfcc_df['feature'].values.tolist() )
testf_df = pd.concat([te_mfcc_df2, te_mfcc_df.label], axis=1)

# tag feature columns with f + num labels
tag_cols(testf_df,'f','label')


# Prepare data (features) for NN and visualize classes (labels)
le = LabelEncoder()

#> replace 'featr' with 'mfcc'
x_train = np.array(tr_mfcc_df.feature.tolist())
y_train = np.array(tr_mfcc_df.label.tolist())
y_train_cat = np_utils.to_categorical(le.fit_transform(y_train))

x_test = np.array(te_mfcc_df.feature.tolist())
y_test = np.array(te_mfcc_df.label.tolist())
y_test_cat = np_utils.to_categorical(le.fit_transform(y_test))

# perform data standardization
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

speakers = le.classes_
num_labels = y_train_cat.shape[1]
num_features = 35

# Create and run a multilayer perceptron 
dnn = Sequential()

# input layer
dnn.add(Dense(128, input_shape=(num_features,)))
dnn.add(Activation('relu'))
dnn.add(Dropout(0.5))

# hidden layer
dnn.add(Dense(64))
dnn.add(Activation('relu'))
dnn.add(Dropout(0.5))

# output layer
dnn.add(Dense(num_labels))
dnn.add(Activation(tf.nn.softmax))
#---

#---
model = dnn
model.summary()

# compile and fit the model
model.compile(loss='categorical_crossentropy', 
              metrics=['accuracy'], optimizer='adam')

print('\nTraining...\n')
model.fit(x_train, y_train_cat,
         batch_size=16, epochs=150)  

# evaluate model accuracy on test set 
te_acc = model.evaluate(x_test, y_test_cat, verbose=False)[1]
print('Test accuracy:', te_acc,'\n') #: 1.0 

# Voice command authentication
# access saved voice command 
clip_path = './Desktop/flask_apps/'
filename = 'audio.wav'
clip = os.path.join(clip_path, filename)

# play real-time voice command
audio_file = audio.from_wav(clip)
print('Playing ... ', clip[-23:])
play(audio_file)

# extract features from real-time audio 
audio_dat, sample_rate = librosa.load(clip, 
                                          res_type='kaiser_fast')  

x_rt = np.mean(librosa.feature.mfcc(y = audio_dat, 
                        sr = sample_rate, 
                        n_mfcc = num_features).T,
                        axis = 0) 

x_rt_t = x_rt.reshape(1, num_features)

# Scale the new set [gives an error with x_rt]
x_rt_t = scaler.transform(x_rt_t)

# predict speaker class or label on real-time audio
rt_class_prob = model.predict(x_rt_t) 
rt_class_prob_rs = rt_class_prob.reshape(len(speakers),1)
rt_class_prob_df = pd.DataFrame(list(map(np.ravel, rt_class_prob_rs)))
rt_class_prob_df.columns = ['f']
rt_class_prob_ls = np.array(rt_class_prob_df['f'].tolist())

# authenticate voice command
prob_thresh = .95
num_prob_thresh = 0
max_prob = np.max(rt_class_prob_ls)
max_prob_val = round(max_prob * 100,2)
max_prob_ind = rt_class_prob_ls.tolist().index(max_prob) 
rt_pred_label = speakers[max_prob_ind]
rt_pred_label = re.sub('\_(.*)','',rt_pred_label)
print('I\'m %s%% sure that you are %s.'\
      % (max_prob_val, rt_pred_label.upper()) )
if max_prob_val > prob_thresh*100:
    print('Authentication completed. You now have access to your car.')
else:
    print('Sorry, I am unable to authenticate. Please repeat your command.')
    
    
    

