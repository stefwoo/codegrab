# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
# from keras import backend as K
import numpy as np


# dimensions of our images.
img_width, img_height = 15, 20

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 543
nb_validation_samples = 260
epochs = 50
batch_size = 8

# num_classes = 10

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

# print "input shape: ", input_shape

# model = Sequential()
# model.add(Dense(512, activation='relu', input_shape=input_shape)
# model.add(Dropout(0.2))
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(10, activation='softmax'))

# model.summary()

# model.compile(loss='categorical_crossentropy',
#               optimizer=RMSprop(),
#               metrics=['accuracy'])


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator()

# this is the augmentation configuration we will use for testing:
# only rescaling
# test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# validation_generator = test_datagen.flow_from_directory(
#     validation_data_dir,
#     target_size=(img_width, img_height),
#     batch_size=batch_size)


m1 = train_generator.next()
print m1
# print len(m1)
# print len(m1[0]),len(m1[1])

# print m1[0][0].shape

print m1[0].shape,m1[1].shape

# print type(m1[0]),type(m1[1])
# for i in m1[0]:
# 	print type(i)
# 	# np.array
# 	print i.shape
# 	# print "-----"


# model.fit_generator(
#     train_generator,
#     steps_per_epoch=nb_train_samples // batch_size,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=nb_validation_samples // batch_size)
