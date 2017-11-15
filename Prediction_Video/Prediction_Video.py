import sys,cv2,os
import numpy as np
os.environ['KERAS_BACKEND']='tensorflow'
sys.path.append("C:/Users/Sai/Desktop/Facial_Expression_Recognition")

import CNN_Model.myVGG as vgg

windowsName = 'Preview Screen'

CASCADE_PATH = "Prediction_Video/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion = ['Angry', 'Fear', 'Happy','Sad', 'Surprise', 'Neutral']
model = vgg.VGG_16('Model_Training/my_model_weights.h5')

capture = cv2.VideoCapture(0)

def grayFace(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    return img_gray

def getFaceCoordinates(image):
    img_gray = grayFace(image)
    rects = cascade.detectMultiScale(
        img_gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(48, 48)
        )
    return rects

def predict_emotion(gray_face):
    resized_img = cv2.resize(gray_face, (48,48), interpolation = cv2.INTER_AREA)
    image = resized_img.reshape(1, 1, 48, 48)
    results = model.predict(image, batch_size=1, verbose=1)
    return results

while True:
    flag, frame = capture.read()
    img_gray = grayFace(frame)
    rects = getFaceCoordinates(frame)
    for (x, y, w, h) in rects:
        face_image = img_gray[y:y+h,x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        results = predict_emotion(face_image)
        cv2.putText(frame, str(emotion[np.argmax(results)]), (x, y),cv2.FONT_HERSHEY_COMPLEX, 1 , (0, 255, 0), 2)
        print (emotion[np.argmax(results)])
        
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
