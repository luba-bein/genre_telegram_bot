import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
from tqdm import tqdm

IMPUT_DIR = '/content/'

data_cvs = pd.read_csv(IMPUT_DIR+'train.csv')
data_cvs = data_cvs.loc[data_cvs['N/A'] != 1]
data_cvs = data_cvs.drop(columns=['N/A'])
data_cvs.head()

classes = data_cvs.columns[2:].tolist()


y = data_cvs.iloc[:,2:]
y = y.to_numpy()

IMG_SIZE = 300
CHANNELS = 3
def read_img(file_name):
    path = IMPUT_DIR+'Images/'+file_name+'.jpg'
    img = image.load_img(path,target_size=(IMG_SIZE,IMG_SIZE,CHANNELS))
    img = image.img_to_array(img)
    img = img/255.0
    return img

X = [read_img(x) for x in tqdm(data_cvs.iloc[:,0])]
X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)


model = Sequential()

model.add(Conv2D(16,kernel_size=(3,3),activation='relu',input_shape=X_train[0].shape))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(len(classes),activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

EPOCHS = 20
history = model.fit(X_train,
                    y_train, 
                    epochs=EPOCHS,
                    validation_data=(X_test,y_test))


model.save('genres_definition.h5')
