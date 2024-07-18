# GATHERING DATA
# import numpy as np
# X_train = np.load('data/X_train.npy')
# y_train = np.load('data/y_train.npy')
# X_test = np.load('data/X_test.npy')
# y_test = np.load('data/y_test.npy')
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
        img = cv2.imread(trainPath + '/' + category + '/{0}'.format(file))
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
        img = cv2.imread(testPath + '/' + category + '/{0}'.format(file))
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
X_train = X_train.reshape((numOfImages, 48, 48, 3))
numOfImages = X_test.shape[0]
X_test = X_test.reshape((numOfImages, 48, 48, 3))

#check
print(X_train.shape)
print(X_train[0])

#convert to categorical
from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Data Augmentation
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=.2,
    zoom_range=.2,
    fill_mode='nearest'
)
datagen.fit(X_train)


#Build the model
#==================

input_shape = X_train.shape[1:]
print(input_shape)


import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, Activation
from tensorflow.keras.layers.experimental.preprocessing import RandomZoom, RandomRotation, RandomFlip, Resizing
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



# defining model
m_net = keras.applications.MobileNetV2(include_top=False, input_shape=(48,48,3), weights='imagenet') #Pre trained
#m_net.trainable = False
base_input = m_net.layers[0].input
base_output = m_net.layers[-1].output
final_output = Flatten()(base_output)
final_output = Dropout(0.5)(final_output)
final_output = Dense(256, kernel_initializer='he_uniform')(final_output)
final_output = Activation('relu')(final_output)
final_output = Dense(128)(final_output)
final_output = Activation('relu')(final_output)
final_output = Dense(7, activation="softmax")(final_output)
model = keras.Model(inputs=base_input, outputs=final_output)

print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=["accuracy"])

batch = 64
epochs = 150
modelFileName = 'emotion_tl.h5'

stepsPerEpoch = np.ceil(len(X_train)/batch)
validationSteps = np.ceil(len(X_train)/batch)

saveBest = keras.callbacks.ModelCheckpoint(modelFileName, monitor='val_loss', save_best_only=True, mode='auto', verbose=1)
stopEarly = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=1)

# train the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=batch),
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    shuffle=True,
                    callbacks=[saveBest, stopEarly])


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
