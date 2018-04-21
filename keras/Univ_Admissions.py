import numpy as np
from keras.utils import np_utils
import tensorflow as tf
# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation

import pandas as pd
data = pd.read_csv('univ_data.csv')
#print(data)

data["gre"] = data["gre"]/800
data["gpa"] = data["gpa"]/4

X = np.array(data)[:,1:]
y = np_utils.to_categorical(np.array(data["admit"]))

model = Sequential()
model.add(Dense(128, input_dim=3))
model.add(Activation('relu'))
#model.add(Dense(32))
#model.add(Activation('relu'))
#model.add(Dense(4))
#model.add(Activation('relu'))

model.add(Dense(2))
model.add(Activation('sigmoid'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

X_train = X
y_train = y

model.fit(X_train, y_train, epochs=1500, batch_size=100, verbose=0)

score = model.evaluate(X_train, y_train)


print("\nAccuracy: ", score[-1])

