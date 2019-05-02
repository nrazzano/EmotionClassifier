#!/usr/bin/env python3

import numpy as np
import sys
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import imutils
import cv2


model = load_model('models/top_layer_model_300e_2.h5', compile=False)
EMOTIONS_ARR_DEBUG = [   0   ,     1    ,    2    ,    3   ,   4  ,     5      ,     6    ]
EMOTIONS_ARR 	   = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']


# TODO: use haarcascade to find face in regular input image
#		for now, just using ROI images
# def detect_face():



i = 0
while(i < 6):
		
	#print('Please give me image...')
	in_filename = input('\nFEED ME AN IMAGE!! ')
	
	# TARGET SIZE MUST BE (48, 48, 1)
	#test_img = load_img(in_filename, target_size=(48, 48))
	#test_img = load_img(in_filename, target_size=(150, 150))
	
	# TODO: resize input image with opencv
	img_raw = cv2.imread(in_filename)	
	img_gray = cv2.imread(in_filename, 0)
	img_roi = cv2.resize(img_raw, (48, 48))
	img_roi = img_roi.astype('float') / 255.0
	
	test_img = img_to_array(img_roi)
	test_img = np.expand_dims(test_img, axis=0)
	result = model.predict(test_img)[0]
	emotion_prob = np.max(result)
	label = EMOTIONS_ARR_DEBUG[result.argmax()]
	print(label)
	#print(result[0][0]) # * 1000)
	i+=1
	#if result[0][0] >= 0.5:
	#	prediction='neutral'
	#else:
	#	prediction='emotion'

	#print(prediction)
	#print(result[0][0])

