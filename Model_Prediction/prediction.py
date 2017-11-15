# some_file.py
import sys
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
    # predict_classes will enable us to select most probable class
    print (emotion[np.argmax(results)])

if __name__ == "__main__":
    main()
