
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import cv2

testPath = 'C:/Users/bruce/Documents/Coding/Datasets/test'

folderList = os.listdir(testPath)
folderList.sort()

X_test = []
y_test = []

#load the testing data into arrays 
folderList = os.listdir(testPath)
folderList.sort()

for i, category in enumerate(folderList):
    files = os.listdir(testPath + '/' + category)
    for file in files:
        print(category+'/'+file)
        img = cv2.imread(testPath + '/' + category + '/{0}'.format(file))
        X_test.append(img)
        y_test.append(i) # each folder will be a number

print('Test Data:')
print(len(X_test))
print(len(y_test))


#convert to numpy
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

#normalize
X_test = X_test / 255.0 

#reshape
numOfImages = X_test.shape[0]
X_test = X_test.reshape((numOfImages, 48, 48, 3))


from tensorflow.keras.utils import to_categorical

y_test = to_categorical(y_test, num_classes=7)

import os

model_path = 'models/'

files = os.listdir(model_path)

model_names = []
models = []

for model_name in files:
    model_names.append(model_name)
    models.append(keras.models.load_model(model_path + model_name))
    
for i in range(len(models)):
    if(model_names[i] == 'emotion.h5') :
        model_score = models[i].evaluate(X_test[:,:,:,:1], y_test)
    else:
        model_score = models[i].evaluate(X_test, y_test)
    print(f'Model: {model_names[i]} has an val accuracy of {model_score}')