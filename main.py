import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

emotion_dict = {0:'ANGRY', 1:'DISGUST', 2:'FEAR', 3:'HAPPY', 4:'NEUTRAL', 5:"SAD", 6:"SURPRISE",}

#import model
model = load_model('models/emotion_tl_1.h5')

#import face detection
alg = 'haarcascade_frontalface_default.xml'
haar_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + alg
)

#get webcam
cap = cv2.VideoCapture(0)
windowResize = 1

def findFaces(image):
    image = cv2.resize(image, (int(image.shape[1]*windowResize), int(image.shape[0]*windowResize)))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = haar_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )
    return faces, gray

def prepareImageForModel(face_img):
    face_img = cv2.resize(face_img, (224, 224), interpolation=cv2.INTER_AREA)
    face_img = cv2.cvtColor(face_img,cv2.COLOR_GRAY2RGB)
    face_img = np.array(face_img).astype('float32')
    face_img = face_img / 255.0
    cv2.imshow('Face Data', cv2.resize(face_img, (480, 480))) # display what the model is seeing
    face_img = np.expand_dims(face_img, axis=0)
    face_img = np.expand_dims(face_img, axis=-1)
    return face_img

lastTime = time.time()
while True:
    t, frame = cap.read()
    faces, gray = findFaces(frame)

    for (x,y,w,h) in faces:
        face_img = gray[y:y+h, x:x+w]
        
        #draw face frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        
        #preprocess face
        face_img = prepareImageForModel(face_img)

        #predict emotion
        resultArray = model.predict(face_img, verbose=0)
        emotion_index = np.argmax(resultArray)
        confidence_value = resultArray[0, emotion_index]
        #print(f'Confidence value of: {emotion} is, {confidence_value} at index: {emotion_index}')
        emotion_label = emotion_dict[emotion_index] + ' ' + str(round(confidence_value, 2))

        #label emotion
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, .4*windowResize, (36, 255, 12), 1)

            
    #fps calculations
    dTime = time.time() - lastTime
    fps = 1/dTime
    lastTime = time.time()
    cv2.putText(frame, f'FPS: {int(fps)}', (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

    #display modified webcam image
    cv2.imshow('Face Landmarks', cv2.resize(frame, (int(frame.shape[1] * (1/windowResize)), int(frame.shape[0] * (1/windowResize)))))

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()