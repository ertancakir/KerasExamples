import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('Iris.csv', header=None)
training_data = data.values

train_X = np.array(training_data[:,0:4], dtype="float32")
data_Y = training_data[:,4]

classes = np.unique(data_Y)
nClasses = len(classes)
dimData = np.prod(train_X.shape[1:])

encoder = LabelEncoder()
encoder.fit(data_Y)
encoded_Y = encoder.transform(data_Y)
train_y = np_utils.to_categorical(encoded_Y)

model = Sequential()
model.add(Dense(4, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(8, activation='tanh'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(train_X, train_y, verbose=1, batch_size=150, nb_epoch=500)

print model.predict(train_X).round()

