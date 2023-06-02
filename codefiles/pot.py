# Using Pandas for Data processing
import pandas as pd
# Using numpy for linear Algebra
import numpy as np
# The OS
import os
import tensorflow as tf
# image processing using keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
# CV2 to read the image
import cv2
# plot the image on a graph
import matplotlib.pyplot as plt

# Specifying the path and Traversing The Data Path
dataset_path = os.path.expanduser('Dataset')
# looping the dataset files
filecount = 0
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        #print(os.path.join(root, file))
        filecount += 1

print(filecount)
data_train = ImageDataGenerator(rescale = 1./255,
                               shear_range =0.2,
                               horizontal_flip = True,
                               validation_split=0.2)

data_training = data_train.flow_from_directory('Dataset',
                                             target_size =(64, 64),
                                             batch_size = 32,
                                             class_mode ='binary',
                                             subset='training')

data_validation = data_train.flow_from_directory('Dataset',
                                             target_size =(64, 64),
                                             batch_size = 32,
                                             class_mode ='binary',
                                             subset='validation')

# Sequential Model Using CNN(convolutional nueral network)
cnn = Sequential([
    # Add the first layer of cnn using convolution filter
    Conv2D(filters=32, kernel_size=5, activation='relu', input_shape=[64, 64, 3]),
    # using the max pooling, with two strides
    MaxPool2D(2, 2),
    #         # Second layer
    #         Conv2D(filters=16, kernel_size=3, activation='relu'),
    #         # using max poling
    #         MaxPool2D(2, 2),
    # Second layer
    Conv2D(filters=32, kernel_size=5, activation='relu'),
    # using max poling
    MaxPool2D(2, 2),

    # Adding the Flatten layer before being fed into the neural network
    Flatten(),
    # one
    # Dense(units=284, activation='relu', name='layer'),
    # Adding One Dense layer of the neural network with 128 neurons
    Dense(units=128, activation='relu'),
    # Lastly the final output layers, using the sigmoid activation
    Dense(units=1, activation='sigmoid')
])

# Now Compiling With The Optimizer Adam
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

# Traning The Model
cnn.fit(x=data_training, validation_data=data_validation, epochs=50)
print(cnn.summary())

