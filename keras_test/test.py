import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


X_train = np.array([2,3,7])
y_train = np.array([2,3,7])
input_dim = X_train.shape[1]

model = Sequential()

model.add(Dense(output_dim=64, input_dim=input_dim))
model.add(Activation("relu"))
model.add(Dense(output_dim=10))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, nb_epoch=5, batch_size=32)