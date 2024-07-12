import cv2
import dlib
import time
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten, Activation
from keras.models import Sequential

emotion_dict = {0:'ANGRY', 1:'HAPPY', 2:'SAD', 3:'SURPRISE', 4:'NEUTRAL'}


def create_model():
    # initialize model
    cnn_model = Sequential()
    # this conv layer has 64 filters! the input shape needs to be the same dimensions of the image
    cnn_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    cnn_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # batch normalization
    cnn_model.add(BatchNormalization())
    # max pooling
    cnn_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    # dropout
    cnn_model.add(Dropout(0.5))

    cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
    cnn_model.add(Dropout(0.25))

    # flatten all the outputs between convolutional and dense layers
    cnn_model.add(Flatten())
    # add a "dense layer" (i.e. the fully connected layers in MLPs) with dropout
    cnn_model.add(Dense(512, activation='relu'))
    # output layer
    cnn_model.add(Dense(5))
    cnn_model.add(Activation('softmax'))

    return cnn_model


#import model
model = create_model() 
model.load_weights('best_cnn_model.h5')

cap = cv2.VideoCapture(0)

windowResize = .6
context_pixels = int(10*windowResize)
face_detector = dlib.get_frontal_face_detector()

dlip_facelandmark = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
lastTime = time.time()
while True:
    t, frame = cap.read()
    frame = cv2.resize(frame, (int(frame.shape[1]*windowResize), int(frame.shape[0]*windowResize)))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray, 1)

    for face in faces:
        (x,y,w,h) = (face.left(), face.top(), face.bottom() - face.top(), face.right() - face.left())
        #draw face frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))
        face_landmarks = dlip_facelandmark(gray, face)
        face_landmarks = [(p.x, p.y) for p in face_landmarks.parts()]
        #draw face landmarks
        for index, p in enumerate(face_landmarks):
            cv2.circle(frame, p, 0, (0, 0, 255), 2)
            #cv2.line(frame, p, face_landmarks[index - 1 if index - 1 >= 0 else len(face_landmarks) - 1], (0,0,255))

        #preprocess face
        face_img = gray[(y-context_pixels):(y+h+context_pixels), (x-context_pixels):(x+w+context_pixels)]
        face_img = cv2.resize(face_img, (48, 48))
        face_img = np.array(face_img).astype('float32')
        face_img = face_img / 255.0
        cv2.imshow('Face Data', cv2.resize(face_img, (480, 480))) # display what the model is seeing
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)


        #predict emotion
        emotion = model.predict(face_img, verbose=0)
        emotion_index = np.argmax(emotion)
        confidence_value = emotion[0, emotion_index]
        print(f'Confidence value of: {emotion} is, {confidence_value} at index: {emotion_index}')
        emotion_label = emotion_dict[emotion_index] + ' ' + str(confidence_value)

        #label emotion
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1*windowResize, (36, 255, 12), 2)

            

    dTime = time.time() - lastTime
    fps = 1/dTime
    lastTime = time.time()
    cv2.putText(frame, f'FPS: {int(fps)}', (0,15), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,0), 2)

    cv2.imshow('Face Landmarks', cv2.resize(frame, (int(frame.shape[1] * (1/windowResize)), int(frame.shape[0] * (1/windowResize)))))

    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()