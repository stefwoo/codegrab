# -*- coding: utf-8 -*-


from __future__ import print_function #兼容python2的print语句

import keras
# from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,Flatten,Activation,MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


# dimensions of our images.
img_width, img_height = 15, 16

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 543
nb_validation_samples = 260
epochs = 12
batch_size = 1

# num_classes = 10
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()


# input=15*20*3, output=15*20*32
model.add(Conv2D(
    nb_filter=32,
    nb_row=2,
    nb_col=2,
    border_mode='same',     # Padding method
    # dim_ordering='th',      # if use tensorflow, to set the input dimension order to theano ("th") style, but you can change it.
    batch_size = batch_size,
    input_shape=input_shape         # channels & height & width
    ))
model.add(Activation('relu'))

# input=15*20*32, output=8*11*32=2816
model.add(MaxPooling2D(
    pool_size=(2, 2),
    strides=(2, 2),
    border_mode='same',    # Padding method
))

# input=8*11*32
model.add(Conv2D(64,(2,2),border_mode="same"))
model.add(Activation("relu"))
# output=7*9*32
model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

# model.add(Conv2D(64,(2,2),border_mode="same"))
# model.add(Activation("relu"))
# # output=7*9*32
# model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))

# model.add(Conv2D(64,(2,2),border_mode="same"))
# model.add(Activation("relu"))
# # output=7*9*32
# model.add(MaxPooling2D(pool_size=(2, 2), border_mode='same'))



# Fully connected layer 1 input shape (64 * 7 * 7) = (3136), output shape (1024)
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# Fully connected layer 2 to shape (10) for 10 classes
model.add(Dropout(0.5))
model.add(Dense(26))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
   rescale=1. / 255,
   shear_range=0.2,
   zoom_range=0.2,
#    horizontal_flip=True
    )

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

import string
s1 = string.lowercase[:]
C1 = [i for i in s1]



train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    classes=C1,
    # color_mode="grayscale",
    batch_size=batch_size)
    # class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    # color_mode="grayscale",
    batch_size=batch_size)
    # class_mode='binary')

# print(train_datagen.class_dictionary)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#保存模型和权重
# import cPickle
# cPickle.dump(model, open('./MyConvNets.pkl',"w"))
json_string = model.to_json()
open('my_model_architecture.json','w').write(json_string)  
model.save_weights('my_model_weights.h5')  




# import keras
# from keras.models import model_from_json  
# import numpy as np
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# model = model_from_json(open('my_model_architecture.json').read())  
# model.load_weights('my_model_weights.h5')  
# img = load_img('2d83aa34-cfef-492a-ba42-ee45fbe7571e.png')
# x = img_to_array(img)
# x = np.rollaxis(x,1)
# x = x.reshape((1,) + x.shape)
# pre_temp1=model.predict_classes(x)
# print(pre_temp1)

# def predit(img):
#     im =  load_img(img)
#     x = img_to_array(im)
#     x = np.rollaxis(x,1)
#     x = x.reshape((1,) + x.shape)
#     pre_temp1=model.predict_classes(x)
#     # print pre_temp1



# from keras.models import model_from_json  

# model = model_from_json(open('my_model_architecture.json').read())  
# model.load_weights('my_model_weights.h5')  



# history = model.fit(x_train, y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(x_test, y_test))
# score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

