#!/usr/bin/env python3

import cv2
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage


def main() :
	print('Starting...')
	
	haar_cascade_face = cv2.CascadeClassifier('/home/nick/Documents/EmotionClassifier/data/haarcascade_frontalface_default.xml')
	input_file = sys.argv[1]
	if not (os.path.exists(input_file)):
		print(input_file + ' DOES NOT EXIST!!!')
		return 0

	input_filename = input_file[input_file.rfind('/')+1:input_file.rfind('.')]
	output_path = os.path.join(os.getcwd(), sys.argv[2])
	if not (os.path.isdir(output_path)):
		print(output_path + ' DOES NOT EXIST!!!')
		return 0
	
	#output_path = os.getcwd()
	print('writing images to: ' + output_path)
	index = 0
	cap = cv2.VideoCapture(sys.argv[1])

	while(cap.isOpened()):
		ret, frame = cap.read()
		if not ret:
			break

		img_roi = detect_faces(haar_cascade_face, frame)
		output_file = input_filename + '_roi_' + str(index) + '.jpg'
		output_full_path = output_path + '/' + output_file
		#print(output_full_path)
		cv2.imwrite(output_path + '/'+ output_file, img_roi) 
		index += 1

	print("done")


# opencv loads images in BGR, so we need to convert to RGB
def convert_to_rgb(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# convert image to grayscale
def convert_to_gray(image):
	return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# resize the image to 32x32
def resize_image(image):
	with open(image, 'r+b') as f:
		with Image.open(f) as im:
			cover = resizeimage.resize_cover(im, [100, 100])
			cover.save('output.png', im.format)


# Draw rect around fact
def draw_face_rect(faces_rects, img_input):
	# TODO: crop image around this rect
	#       also make them all uniform size?
	for(x,y,w,h) in faces_rects:
		#cv2.rectangle(img_input, (x, y), (x+w, y+h), (0, 255, 0), 2)
		img_roi = img_input[y:y+h, x:x+w]
		return img_roi


# use cascade to detect face in image(frame)
def detect_faces(cascade, img_input, scale_factor = 1.1):
	img_copy = img_input.copy()

	img_gray = convert_to_gray(img_input)
	faces_rects = cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
	return draw_face_rect(faces_rects, img_gray)
	#print('Faces found: ', len(faces_rects))


if __name__ == '__main__' :
	main()
