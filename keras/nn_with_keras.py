#!/usr/bin/env python3

# Nick's Convolutional NN using Keras and TensorFlow backend

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Dense(32, input_shape=(784,)),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])

# initialize the convolutional neural net
classifier = Sequential()

# add layers; Convolution, pooling, flattening respectively
classifier.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())

# Connecting the cnn to a nn; nn will be 2 layers, one with a sigmoid function
# so we can find the probability of the emotion being neutral or not
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_data = ImageDataGener
