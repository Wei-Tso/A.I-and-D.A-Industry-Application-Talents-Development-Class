# -*- coding: utf-8 -*-
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


#####################
# 將訓練好的權重載入
#####################
model_car.load_weights("./car_train.h5")


#####################
# 將 carlogosample.csv 載入
# 對所有的訓練資料進行預測，將結果寫入 csv
# 看是否跟原始的 label 資料(carlogo.csv)相同
#####################
result_df = pd.read_csv('carlogosample.csv')


for index , row in result_df.iterrows() :
    filename = row['image_id']
    img = cv2.imread("./carlogomap/" + filename + ".jpg", cv2.IMREAD_COLOR)
    img = cv2.resize(img , (img_size , img_size))
    x_test = img.reshape(1 , img_size , img_size , 3).astype('float32')
    x_test = x_test / 255

    prediction = model_car.predict_classes(x_test).ravel()[0]
    print(filename , prediction)
    
    if prediction == 0:
        result_df.iloc[index,1] = 1
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 1:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 1
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 2:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 1
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 3:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 1
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 4:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 1
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 5:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 1
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 6:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 1
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 7:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 1
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 8:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 1
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 9:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 1
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 0
    elif prediction == 10:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 1
        result_df.iloc[index,12] = 0
    elif prediction == 11:
        result_df.iloc[index,1] = 0
        result_df.iloc[index,2] = 0
        result_df.iloc[index,3] = 0
        result_df.iloc[index,4] = 0
        result_df.iloc[index,5] = 0
        result_df.iloc[index,6] = 0
        result_df.iloc[index,7] = 0
        result_df.iloc[index,8] = 0
        result_df.iloc[index,9] = 0
        result_df.iloc[index,10] = 0
        result_df.iloc[index,11] = 0
        result_df.iloc[index,12] = 1
        

#####################
# 將結果輸出
#####################               
result_df.to_csv('submission.csv' , index = False)

