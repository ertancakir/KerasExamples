import numpy as np
import cv2
import glob
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D, Dropout, Flatten

from sklearn.preprocessing import LabelEncoder # for one hot encoding

train_data = []

data = pd.read_csv('output.csv', header=None)
output_data = data.values

encoder = LabelEncoder()
encoder.fit(output_data)
encoded_output_data = encoder.transform(output_data)
output_data_one_hot = np_utils.to_categorical(encoded_output_data)

files = glob.glob("./letters/*.jpeg")

for file in files:
    image = cv2.imread(file)
    train_data.append(image)

train_data = np.array(train_data)


dimData = np.prod(train_data.shape[1:])
train_data = train_data.astype('float32')
train_data /= 255


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(512, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_data, output_data_one_hot, batch_size=256, epochs=300, verbose=1)


test_data_images = []

files = glob.glob("./test/*.jpeg")

for file in files:
    image = cv2.imread(file)
    image = cv2.resize(image,(115,74))
    test_data_images.append(image)

test_data = np.array(test_data_images,dtype = 'float32')

print model.predict(test_data).round()