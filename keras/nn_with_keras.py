#!/usr/bin/env python3

# Nick's Convolutional NN using Keras and TensorFlow backend

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout


print("hello")


model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# initialize the convolutional neural net
classifier = Sequential()

# add layers; Convolution, pooling, flattening respectively
classifier.add(Conv2D(32, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

# Connecting the cnn to a nn; nn will be 2 layers, one with a sigmoid function
# so we can find the probability of the emotion being neutral or not
classifier.add(Dense(activation = 'relu', units=128))
classifier.add(Dense(activation = 'sigmoid', units=1))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fit the CNN to the images
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

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
		'dataset/train1',
		target_size = (150, 150),
		batch_size = 16,
		class_mode = 'binary')

#print('\n\nSomething2\n\n')
test_set = test_datagen.flow_from_directory(
		'dataset/test1',
		target_size = (150, 150),
		batch_size = 16,
		class_mode = 'binary')

#print('\n\nSomething3\n\n')

#from IPython.display import display
#from PIL import Image

classifier.fit_generator(
		training_set,
		steps_per_epoch = 8000,
		epochs = 1,
		validation_data=test_set,
		validation_steps = 800)

classifier.save_weights('second_try_1epoch.h5')

