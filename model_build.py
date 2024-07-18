
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
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
    for file in files:
        print(category+'/'+file)
        img = cv2.imread(trainPath + '/' + category + '/{0}'.format(file), 0)
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
    for file in files:
        print(category+'/'+file)
        img = cv2.imread(testPath + '/' + category + '/{0}'.format(file), 0)
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
X_train = X_train.reshape((numOfImages, 48, 48, 1))
numOfImages = X_test.shape[0]
X_test = X_test.reshape((numOfImages, 48, 48, 1))

#check
print(X_train.shape)
print(X_train[0])

#convert to categorical
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Data Augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
)
datagen.fit(X_train)

#Build the model
#==================

input_shape = X_train.shape[1:]
print(input_shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

physical_devices = tf.config.list_physical_devices("GPU")
print(len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# defining model
model = Sequential()
model.add(Conv2D(32, input_shape=input_shape, kernel_size=(3,3), padding='same', activation="relu"))
model.add(keras.layers.BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(128, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(512, kernel_size=(3,3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dense(7, activation='softmax'))

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

batch = 32
epochs = 60
modelFileName = 'emotion.h5'

stepsPerEpoch = np.ceil(len(X_train)/batch)
validationSteps = np.ceil(len(X_train)/batch)

saveBest = keras.callbacks.ModelCheckpoint(modelFileName, monitor='val_loss', save_best_only=True, mode='auto', verbose=1)

# train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch),
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=[saveBest])

# show the result on pyplot
acc = history.history['accuracy']
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(len(acc))

#show the charts
plt.plot(epochs, acc, 'r', label='Train accurcy')
plt.plot(epochs, val_acc, 'b', label='Val accurcy')
plt.xlabel('Epocs')
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend(loc='lower right')
plt.show()

#show loss charts

plt.plot(epochs, loss, 'r', label='Train loss')
plt.plot(epochs, val_loss, 'b', label='Val loss')
plt.xlabel('Epocs')
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend(loc='lower right')
plt.show()
