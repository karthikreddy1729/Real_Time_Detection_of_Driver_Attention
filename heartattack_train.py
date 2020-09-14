# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 12:30:14 2020

@author: karth
"""

import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_input = "./dataset/train/"
train_val = "./dataset/test/"

batchsize = 48
n_classes = 2   # Don't chage, 0=Infarct, 1=Normal

train_gen = train_datagen.flow_from_directory(
    train_input,
    target_size=(256, 256),
    batch_size=batchsize,
    class_mode='categorical')
val_gen = val_datagen.flow_from_directory(
    train_val,
    target_size=(256, 256),
    batch_size=batchsize,
    class_mode='categorical')

model = Sequential()
model.add(Conv2D(128, kernel_size=(3 ,3), padding ="same", input_shape=(256, 256, 3), activation='relu'))
model.add(MaxPooling2D((3,3), strides=(3,3)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((3,3), strides=(3,3)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Conv2D(1024, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(n_classes, activation='softmax'))

sgd = SGD(lr=0.003  , decay=1e-4, momentum=0.9, nesterov=True)

model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['acc'])


n_epochs = 20
n_steps = 500

# for i in range(10):    
#     model.fit_generator(
#         train_gen,
#         steps_per_epoch=n_steps,
#         epochs=n_epochs,
#         validation_data=val_gen,
#         validation_steps=100,
#         verbose=1
#     )   
    
#     model.save("model/model3.h5",overwrite=True,include_optimizer=True)
    
model.fit_generator(
        train_gen,
        steps_per_epoch=n_steps,
        epochs=n_epochs,
        validation_data=val_gen,
        validation_steps=100,
        verbose=1
    )   
    
model.save("model/model3.h5",overwrite=True,include_optimizer=True)



