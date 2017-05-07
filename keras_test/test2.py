from keras.layers import Dense, Activation
from keras.models import Sequential
model = Sequential([
    Dense(32, input_dim=3),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd')

import numpy as np

# x = np.ndarray([[0,1,3], [1,1,2], [1,0,1]])
# print x.shape
model.fit([[0,1,3], [1,1,2], [1,0,1]], [1,2,3])