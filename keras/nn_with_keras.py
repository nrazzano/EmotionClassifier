#!/usr/bin/env python3

# Nick's Convolutional NN using Keras and TensorFlow backend

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import Adamax


'''
def get_pixels(in_pixels):
	img_size=48
'''



'''
# initialize the convolutional neural net
classifier = Sequential()

# add layers; Convolution, pooling, flattening respectively
classifier.add(Conv2D(32, (3, 3), input_shape = (48, 48, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

# Connecting the cnn to a nn; nn will be 2 layers, one with a sigmoid function
# so we can find the probability of the emotion being neutral or not
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dense(activation = 'sigmoid', units=1))
'''

classifier=Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (48, 48, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(activation='relu', units=128))
#classifier.add(Dense(256, input_shape=(256,), activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(128, input_shape=(256,)))
classifier.add(Dense(7, activation='softmax'))

adamax = Adamax()

classifier.compile(loss = 'categorical_crossentropy',
					optimizer=adamax, metrics = ['accuracy'])

# Fit the CNN to the images
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#pixel_arr=get_pixels(raw_data[['pixels']])






train_datagen = ImageDataGenerator(
		rescale = 1./255,
		shear_range = 0.2,
		zoom_range = 0.2,
		horizontal_flip = True)

#img = load_img('dataset/train1/emotions/nick_emotions_roi_205.jpg')
#x = img_to_array(img)
#x = x.reshape((1,) + x.shape)

#i = 0
#for batch in train_datagen.flow(x, batch_size=1,
#							save_to_dir='preview', save_prefix='emo', save_format='jpeg'):
#	i += 1
#	if i > 20:
#		break


test_datagen = ImageDataGenerator(rescale = 1./255)
#print('\n\nSomethingi1\n\n')
training_set = train_datagen.flow_from_directory(
		'dataset/Training',
		target_size = (48, 48),
		batch_size = 32,
		class_mode = 'categorical')

#print('\n\nSomething2\n\n')
test_set = test_datagen.flow_from_directory(
		'dataset/PrivateTest',
		target_size = (48, 48),
		batch_size = 32,
		class_mode = 'categorical')


#from IPython.display import display
#from PIL import Image


classifier.fit_generator(
		training_set,
		steps_per_epoch = 8000,
		epochs = 10,
		validation_data=test_set,
		validation_steps = 800)

classifier.save_weights('weights_5_100epoch.h5')
classifier.save('classifier_model_5_100epoch.h5')

