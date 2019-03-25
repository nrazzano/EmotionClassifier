#!/usr/bin/env python3
import numpy as np
import cv2 as cv
import sys
import matplotlib.pyplot as plt
from PIL import Image
from resizeimage import resizeimage


def main():
	print("hello")

	img_raw = get_image()

	haar_cascade_face = cv.CascadeClassifier('data/haarcascade_frontalface_default.xml')

	detect_faces(haar_cascade_face, img_raw)

	cv.imwrite('rect_over_face.png', img_raw)

	print("done")


def get_image():
	return cv.imread(sys.argv[1])


# opencv loads images in BGR, so we need to convert to RGB
def convert_to_rgb(image):
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


# convert image to grayscale
def convert_to_gray(image):
	return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


# resize the image to 32x32
def resize_image(image):
	with open(image, 'r+b') as f:
		with Image.open(f) as im:
			cover = resizeimage.resize_cover(im, [100, 100])
			cover.save('output.png', im.format)


def detect_faces(cascade, img_input, scale_factor = 1.1):
	img_copy = img_input.copy()

	img_gray = convert_to_gray(img_input)

	faces_rects = cascade.detectMultiScale(img_gray, scaleFactor = 1.2, minNeighbors = 5)
	print('Faces found: ', len(faces_rects))

	for(x,y,w,h) in faces_rects:
		cv.rectangle(img_input, (x, y), (x+w, y+h), (0, 255, 0), 2)


if __name__ == "__main__":
	main()
