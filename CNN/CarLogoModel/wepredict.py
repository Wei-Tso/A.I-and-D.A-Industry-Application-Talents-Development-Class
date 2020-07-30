import cv2
import pandas as pd
from keras.models import Sequential  
from keras.layers import Dense , Dropout , Flatten , Conv2D , MaxPooling2D

img_size = 64

def build_model():
    model = Sequential()
    model.add(Conv2D(filters = 32 , kernel_size = (3 , 3) , input_shape = (64 , 64 , 3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(filters = 32 , kernel_size = (3 , 3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2 , 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters = 64 , kernel_size = (3 , 3) , activation = 'relu' , padding = 'same'))
    model.add(Conv2D(filters = 64 , kernel_size = (3 , 3) , activation = 'relu' , padding = 'same'))
    model.add(MaxPooling2D(pool_size = (2 , 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(128 , activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64 , activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(12 , activation = 'softmax'))

    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['acc'])
    
    return model

model_car = build_model()
model_car.load_weights("./car_train.h5")


#####################
# 載入任一張圖片，進行預測
#####################
img = cv2.imread("./predictPIC/test9_1.jpg" , cv2.IMREAD_COLOR)
img = cv2.resize(img , (img_size , img_size))
x_test = img.reshape(1,img_size , img_size , 3).astype('float32')
x_test = x_test / 255

prediction = model_car.predict_classes(x_test).ravel()[0]
print(prediction)