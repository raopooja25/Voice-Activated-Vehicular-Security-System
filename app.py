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
from keras.preprocessing.image import array_to_img, img_to_array, load_img,\
									  ImageDataGenerator
from keras.layers import AveragePooling2D, BatchNormalization, Conv2D, \
						 MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.models import load_model
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle as sh
import os
import re
from sklearn.linear_model import LogisticRegression


app = flask.Flask(__name__)
img_width, img_height = 150, 102
n_tr_samples = 750
n_vl_samples = 150
n_te_samples = 10  
n_epochs = 10
input_tensor = (img_width, img_height, 3)
prob_thresh = .95 
num_prob_thresh = 0 
floor = 10  
num_features = 25

@app.route('/')
def index():
	return flask.render_template(template_name_or_list="index.html")
app.add_url_rule(rule="/", endpoint="homepage", view_func=index)

@app.route('/process', methods=['POST'])
def record_audio():
	if request.method == 'POST':
		#RECORDS AUDIO
		r = speech_recognition.Recognizer()
		mic = speech_recognition.Microphone()
		with mic as source:
			r.adjust_for_ambient_noise(source)
			audio_f = r.listen(source, phrase_time_limit = 8)
		#SAVES AUDIO
		with open("audio.wav", "wb") as f:
			f.write(audio_f.get_wav_data())
		#SPECTROGRAM
		y, sr = librosa.load('audio.wav')
		# make and display a mel-scaled power spectrogram
		S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
		# Convert to log scale (dB). Use the peak power as reference.
		log_S = librosa.power_to_db(S)
		fig = plt.figure(figsize=(12,4))
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		fig.add_axes(ax)
		# Display the spectrogram on a mel scale        
		# sample rate and hop length parameters are used to render the time axis
		librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
		# Make the figure layout compact
		plt.savefig('wav.jpg')
		plt.close()
		#message()
		#predict()
		#return "Upload complete."
	else:
		return "Upload incomplete."

def message():
	return "Upload complete."
	
@app.route('/predict', methods=['POST'])
def predict():
	try:
		# AUDIO TRANSCRIPT
		txt = transcript()
		# SPEAKER, TEXT PREDICTIO
		pred_lr_label, pred_lr_max_prob_val = lr_prediction("audio.wav") #Multinomial logistic regression model
		pred_cnn_label, pred_cnn_max_prob_val = cnn_prediction("audio.wav") #CNN model
		return pred_cnn_label + "" + pred_cnn_max_prob_val
		if pred_cnn_label == pred_lr_label:
			mod_dict = {pred_lr_label: max(pred_cnn_max_prob_val, pred_lr_max_prob_val)}
		else:
			mod_dict = {pred_cnn_label: pred_cnn_max_prob_val, pred_lr_label: pred_lr_max_prob_val}
		return mod_dict #PLEASE FIX THIS AND REDIRECT TO RESULT.HTML
		#return pred_lr_label + txt
	except:
		return "ERROR. Please try recording once again."

def transcript():
	#return "H", "H" 
	r = speech_recognition.Recognizer()
	audio_temp = speech_recognition.AudioFile('audio.wav')
	with audio_temp as source:
		audio_f = r.record(source)
	txt = r.recognize_google(audio_f)
	return txt

def lr_prediction(clip):
	"""
	"""
	lr, speakers, scaler = None, None, None
	if not os.path.isfile("audio.wav"):
		return "Please record audio before requesting predictions", "ERROR"
	#return "H", "H" #Check
	if os.path.isfile("lr.pickle"):
		with open("lr.pickle", "rb") as handle:
			lr = pickle.load(handle)
		with open("speakers.pickle", "rb") as handle:
			speakers = pickle.load(handle)
		with open("scalar.pickle", "rb") as handle:
			scaler = pickle.load(handle)
	else:
		tr_mfcc = np.load('25_features_mfcc_train.npy')
		tr_mfcc_df = pd.DataFrame(list(map(np.ravel, tr_mfcc)))
		tr_mfcc_df.columns = ['feature','label']
		tr_mfcc_df2 = pd.DataFrame(tr_mfcc_df['feature'].values.tolist() )
		trainf_df = pd.concat([tr_mfcc_df2, tr_mfcc_df.label], axis=1)
		#return "H", "H" 
		tag_cols(trainf_df,'f','label') 
		# create df of training data features and labels from saved features file.
		te_mfcc = np.load('25_features_mfcc_test.npy')
		te_mfcc_df = pd.DataFrame(list(map(np.ravel, te_mfcc)))
		te_mfcc_df.columns = ['feature','label']
		te_mfcc_df2 = pd.DataFrame(te_mfcc_df['feature'].values.tolist() )
		testf_df = pd.concat([te_mfcc_df2, te_mfcc_df.label], axis=1)
		tag_cols(testf_df,'f','label')
		# prepare data
		#Check
		le = LabelEncoder()
		x_train = np.array(tr_mfcc_df.feature.tolist())
		y_train = np.array(tr_mfcc_df.label.tolist())
		y_train_cat = np_utils.to_categorical(le.fit_transform(y_train))
		x_test = np.array(te_mfcc_df.feature.tolist())
		y_test = np.array(te_mfcc_df.label.tolist())
		y_test_cat = np_utils.to_categorical(le.fit_transform(y_test))
		scaler = StandardScaler().fit(x_train)
		x_train = scaler.transform(x_train)
		x_test = scaler.transform(x_test)
		#global speakers 
		speakers = le.classes_
		
		num_labels = y_train_cat.shape[1]
		lr = LogisticRegression(solver='lbfgs', multi_class='auto')
		lr.fit(x_train, y_train)
		pickle.dump(lr, open("lr.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
		pickle.dump(speakers, open("speakers.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
		pickle.dump(scaler, open("scalar.pickle", "wb"), protocol=pickle.HIGHEST_PROTOCOL)
	"""
	Required parameters for below:
	clip
	num_features 
	speakers 
	"""
	audio_dat, sample_rate = librosa.load(clip, 
									  res_type='kaiser_fast')  
	x_rt = np.mean(librosa.feature.mfcc(y = audio_dat, 
							sr = sample_rate, 
							n_mfcc = num_features).T,
							axis = 0) 
	x_rt_t = x_rt.reshape(1, num_features) #> need before rescaling
	x_rt_t = scaler.transform(x_rt_t)
	pred_lr_class = lr.predict(x_rt_t)[0] 
	pred_lr_prob = lr.predict_proba(x_rt_t)
	pred_lr_prob_rs = pred_lr_prob.reshape(len(speakers),1)
	pred_lr_prob_df = pd.DataFrame(list(map(np.ravel, pred_lr_prob_rs)))
	pred_lr_prob_df.columns = ['f']
	pred_lr_prob_ls = np.array(pred_lr_prob_df['f'].tolist())
	pred_lr_max_prob = np.max(pred_lr_prob_ls)
	pred_lr_max_prob_val = round(pred_lr_max_prob * 100,2)
	pred_lr_max_prob_idx = pred_lr_prob_ls.tolist().index(pred_lr_max_prob)
	pred_lr_label = speakers[pred_lr_max_prob_idx]
	pred_lr_label = re.sub('\_(.*)','', pred_lr_label)
	return pred_lr_label, pred_lr_max_prob_val

def cnn_prediction(clip):
	if not os.path.isfile("audio.wav"):
		return "Please record audio before requesting predictions", "ERROR"
	trn_dir = '/Users/Owner/Alexa_Project_IEOR290/deployment/final/nn_audio/trn'
	val_dir = '/Users/Owner/Alexa_Project_IEOR290/deployment/final/nn_audio/val'
	tst_dir = '/Users/Owner/Alexa_Project_IEOR290/deployment/final/nn_audio/tst'
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
	pred_imgs = []
	pred_dir = 'C:/Users/Owner/Alexa_Project_IEOR290/deployment/final/nn_audio/chk'
	pred_imgs = []
	return "H", "H"
	for files in os.listdir(pred_dir + '/' + 'speaker'):
		if files.endswith('.jpg'):
			pred_imgs.append(files)
	num_pred = len(pred_imgs)
	te_gen = datagen.flow_from_directory(
					directory=pred_dir,
					target_size=(img_width, img_height),
					#color_mode='rgb',
					batch_size=batch_size,
					class_mode=None,
					shuffle=False)
	te_gen.reset()
	pred_cnn = cnn.predict_generator(te_gen, verbose=1, 
									 steps=num_pred/batch_size)
	pred_cnn_rs = pred_cnn.reshape(len(speakers),1)
	pred_cnn_df = pd.DataFrame(list(map(np.ravel, pred_cnn_rs)))
	pred_cnn_df.columns = ['f']
	pred_cnn_ls = np.array(pred_cnn_df['f'].tolist())
	pred_cnn_max_prob = np.max(pred_cnn_ls)
	pred_cnn_max_prob_val = round(pred_cnn_max_prob * 100,2)
	pred_cnn_max_prob_idx = pred_cnn_ls.tolist().index(pred_cnn_max_prob) 
	pred_cnn_label = speakers[pred_cnn_max_prob_idx]
	pred_cnn_label = re.sub('\_(.*)','', pred_cnn_label)
	return pred_cnn_label, pred_cnn_max_prob_val

#UTILITY
def plot_loss_acc(hist):
	f, ax = plt.subplots()
	ax.plot([None] + hist.history['acc'], 'o-', c='r')
	ax.plot([None] + hist.history['val_acc'], '*-', c='g')
	ax.legend(['Train acc', 'Val acc'], loc = 0)
	ax.set_title('Training/Validation Accuracy per Epoch')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Accuracy') 
	plt.plot()
	
	f, ax = plt.subplots()
	ax.plot([None] + hist.history['loss'], 'o-', c='r')
	ax.plot([None] + hist.history['val_loss'], '*-', c='g')
	ax.legend(['Train loss', 'Val loss'], loc = 0)
	ax.set_title('Training/Validation Loss per Epoch')
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Loss') 
	plt.plot()
	
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

if __name__ == '__main__':
#   webbrowser.open('http://localhost:5000')
	app.run(host="localhost", port=8888, debug=True)

