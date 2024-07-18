import os
import numpy as np
import cv2

trainPath = 'C:/Users/bruce/Documents/Coding/Datasets/train'
testPath = 'C:/Users/bruce/Documents/Coding/Datasets/test'

folderList = os.listdir(trainPath)
folderList.sort()
print(folderList)

X_train = []
y_train = []

X_test = []
y_test = []

# load the training data into arrays

for i, category in enumerate(folderList):
    files = os.listdir(trainPath + '/' + category)
    file_amount = len(files)
    for file in files[:int(file_amount/2)]:
        print(category+'/'+file)
        img = cv2.imread(trainPath + '/' + category + '/{0}'.format(file))
        img = cv2.resize(img, (224, 224))
        X_train.append(img)
        y_train.append(i) # each folder will be a number

print(len(X_train))

#show first image
# img1 = X_train[0]
# cv2.imshow('img1', img1)
# cv2.waitKey(0)

# check the labels 
print(y_train)
print(len(y_train))

#load the testing data into arrays 
folderList = os.listdir(testPath)
folderList.sort()

for i, category in enumerate(folderList):
    files = os.listdir(testPath + '/' + category)
    file_amount = len(files)
    for file in files[:int(file_amount/2)]:
        print(category+'/'+file)
        img = cv2.imread(testPath + '/' + category + '/{0}'.format(file))
        img = cv2.resize(img, (224, 224))
        X_test.append(img)
        y_test.append(i) # each folder will be a number

print('Test Data:')
print(len(X_test))
print(len(y_test))


#convert to numpy
X_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
X_test = np.array(X_test, 'float32')
y_test = np.array(y_test, 'float32')

#check
print(X_train.shape)
print(X_train[0])

#normalize
X_train = X_train / 255.0
X_test = X_test / 255.0 

#reshape
numOfImages = X_train.shape[0]# 28709
X_train = X_train.reshape((numOfImages, 224, 224, 3))
numOfImages = X_test.shape[0]
X_test = X_test.reshape((numOfImages, 224, 224, 3))

#check
print(X_train.shape)
print(X_train[0])

#convert to categorical
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)



np.save('data/X_train', X_train)
np.save('data/y_train', y_train)
np.save('data/X_test', X_test)
np.save('data/y_test', y_test)