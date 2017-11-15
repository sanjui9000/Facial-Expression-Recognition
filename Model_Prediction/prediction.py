import sys
from gtts import gTTS
import os
from pygame import mixer
from pygame import time

# Tensorflow specific fix - CUDNN Error
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

os.environ['KERAS_BACKEND']='tensorflow'

sys.path.append('C:/Users/Sai/Desktop/Facial_Expression_Recognition/CNN_Model')
sys.path.append('C:/Users/Sai/Desktop/Facial_Expression_Recognition/Model_Training')

emotion = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']

import myVGG
model = myVGG.VGG_16('C:/Users/Sai/Desktop/Facial_Expression_Recognition/Model_Training/my_model_weights.h5')

import cv2
import numpy as np

def preprocessing(img, size=(48, 48)):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, size).astype(np.float32)
    return img

def predict_emotion(gray_face):
    resized_img = cv2.resize(gray_face, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    results = model.predict(image, batch_size=1, verbose=1)
    return results

def main():
    print ('Image Prediction Mode')
    img = preprocessing(cv2.imread('C:/Users/Sai/Desktop/Facial_Expression_Recognition/Model_Prediction/Images/angry.jpg'))
    results = predict_emotion(img)
    tts = gTTS(text='Expression is ... {}'.format(emotion[np.argmax(results)]), lang='en', slow=False)
    tts.save("Expressions.mp3")

    mixer.init()
    mixer.music.load('Expressions.mp3')
    mixer.music.play()
    while mixer.music.get_busy():
        time.Clock().tick(30)
    else:
        mixer.music.play(-1)
        mixer.quit()
        os.remove('Expressions.mp3')
    print (emotion[np.argmax(results)])

if __name__ == "__main__":
    main()