import flask
from flask import Flask, render_template, request
import werkzeug
import pickle
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import librosa.display
import pandas as pd
import numpy as np
import pyaudio
from pydub import AudioSegment as audio
import struct
import wave
from scipy.fftpack import fft
#from recorder import Recorder, RecordingFile
import speech_recognition
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
import os
import re

def model():
  # define dir paths
  trn_dir = './Desktop/nn_audio/trn/'
  val_dir = './Desktop/nn_audio/val/'
  tst_dir = './Desktop/nn_audio/tst/'
  # get speakers/labels
  val_imgs = []
  speakers = [] 
  for path,subdirs,files in os.walk(val_dir):
      for subdir in subdirs:
          speakers.append(subdir)
          folder_path = os.path.join(val_dir + subdir + '/')
          folder_path = ''.join(folder_path)
          for files in os.listdir(folder_path):
              if files.endswith('.jpg'):
                  val_imgs.append(files)
  speakers = sorted(speakers)
  # Set up for cnn prediction
  # Read images in dirs and generate batches of image data
  datagen = ImageDataGenerator(rescale=1./255)
  tr_gen = datagen.flow_from_directory(
              trn_dir,                             
              target_size=(img_width, img_height),
              batch_size=32,
              class_mode='categorical',
              shuffle=False) 
  tr_lab = tr_gen.classes
  out = len(np.unique(tr_lab)) 
  vl_gen = datagen.flow_from_directory(
              val_dir,                            
              target_size=(img_width, img_height),
              batch_size=16,
              class_mode='categorical',
              shuffle=False) 
  vl_lab = vl_gen.classes
  vl_jump = int(len(vl_lab)/out)
  te_gen = datagen.flow_from_directory(
              tst_dir,                            
              target_size=(img_width, img_height),
              batch_size=2,
              class_mode='categorical',   
              shuffle=False) 
  te_lab = te_gen.classes
  te_jump = int(len(te_lab)/out)
  # create and run cnn model
  cnn = Sequential()
  cnn.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu', 
                   input_shape=input_tensor))
  cnn.add(AveragePooling2D())
  cnn.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
  cnn.add(AveragePooling2D())
  cnn.add(Flatten())
  cnn.add(Dense(units=120, activation='relu'))
  cnn.add(Dense(units=84, activation='relu'))
  cnn.add(Dense(out))
  cnn.add(Activation(tf.nn.softmax))
  cnn.summary()
  # Compile and run the model
  cnn.compile(loss='categorical_crossentropy', 
                optimizer='adam', 
                metrics=['accuracy'])
  hist_cnn = cnn.fit_generator(
              tr_gen,                  
              steps_per_epoch=100,
              epochs=n_epochs,         
              validation_data=vl_gen,  
              validation_steps=10)
  plot_loss_acc(hist_cnn)
