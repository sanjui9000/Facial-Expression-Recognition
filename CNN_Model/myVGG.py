#Sequential package to initialize our Neural Network
from keras.models import Sequential
#Flatten is used for step 4, i.e.convert pooled feature maps into large feature vector that becomes input of our fully connected network.
#Dense is used to add fully connected layers and classic ANN
from keras.layers.core import Flatten, Dense, Dropout
#Convolution2D is used for completing first step of CNN, i.e. adding convolution layers
#2D because they are images.Note: Videos are in 3D.
#MaxPooling2D is used for completing step 3, i.e. add pooling layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

def VGG_16(weights_path=None, shape=(48, 48)):
    # Create object of Sequential class to initialize CNN
    model = Sequential()
    
    # It's easier to design networks if you preserve height and width and don't have to worry too much about tensor dimensions when going from one layer to another because dimensions will just "work".
    model.add(ZeroPadding2D((1,1), input_shape=(1, 48, 48)))
    
    # Number of filters i.e. first parameter is number of feature maps we want to use(One feature map created for each filter used).
    # Number of filters argument also requires number of rows and columns in feature detector.
    # For the next statement, 32 is number of filters(Feature detectors) you want to use.
    # Usually we start with 32, then go further with 64,128.. 256 in future layers
    # 3, 3 is number of rows and number of columns for Feature Detector.
    # Activation function is relu as per step 1.
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    # pool_size is generally (2,2). Will reduce size of feature maps and divide it by 2.
    # In short, we just reduced complexity of our model without affecting performance.
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    # Take all pooled feature maps and put in into one huge vector(Spatial information preserved)
    model.add(Flatten())
    
    # Dense is used to add fully connected layer.
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='softmax'))
    
    print ("Create model successfully")
    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer='adam', loss='categorical_crossentropy', \
        metrics=['accuracy'])

    return model
