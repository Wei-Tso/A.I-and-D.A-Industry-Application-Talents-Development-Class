import numpy as np
import cardataset as ds
from sklearn.utils import class_weight
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten , Conv2D , MaxPooling2D
from sklearn.model_selection import train_test_split

try:
    x = np.load("train_x_aug.npy")
    y = np.load("train_y_aug.npy")
    
    # x = np.load("train_x.npy")
    # y = np.load("train_y.npy")
except: 
    x , y = ds.load_data_augmentation()
    # x , y = ds.load_data()


#####################
# 測試程式
# 將圖片較多的4組另外存
#####################
'''
x = x[np.where((y==0) | (y==3) | (y==4) | (y==7))]
y = y[np.where((y==0) | (y==3) | (y==4) | (y==7))]
y[np.where(y==3)] = 1
y[np.where(y==4)] = 2
y[np.where(y==7)] = 3

import cv2
c = 0
for i in range(len(x)):
    cv2.imwrite('tmp/' + str(y[i]) + '/' + str(c) + '.jpg' , x[i].astype('uint8'))
    c += 1 
'''


#####################
# 自己創建 shuffle 函數
# 這樣之後的 history = model.fit() 就可以不用下 shuffle=True
#####################
import random
xy = list(zip(x , y))
random.shuffle(xy)
x , y = zip(*xy)

x = np.array(x)
y = np.array(y)


#####################
# 因為在 train_test_split() 有下 stratify = y
# 因此不需要 class_weight.compute_class_weight()
#####################
'''
class_weights = class_weight.compute_class_weight('balanced' , np.unique(y) , y)
print(class_weights)
'''


#####################
# 將資料集切分為 train set 和 test set  -->> x_train , x_test , y_train , y_test
# 測試集大小 test_size = 0.2
# stratify = y
#####################
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , stratify = y)


#####################
# 將 y_train、y_test 進行 One-hot Encoding
#####################
y_train = to_categorical(y_train, 12)
y_test = to_categorical(y_test, 12)


#####################
# 將 x_train、x_test 進行正規化
#####################
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0


y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

print(x_train.shape , y_train.shape)
print(x_test.shape , y_test.shape)

####################################################################################


#####################
# 建立模型
#####################
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

print(model.summary())

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['acc'])


#####################
# 將 x_test 以及 y_test 當作 validation data
# 以 tuple 傳入
#####################
history = model.fit(x_train , y_train , batch_size = 128 , epochs = 50 , validation_data = (x_test , y_test) , verbose = 1)


#####################
# 畫出 accuracy 曲線
#####################
import matplotlib.pyplot as plt
def show_train_history(train_history):
    plt.plot(train_history.history['acc'])
    plt.plot(train_history.history['val_acc'])
    plt.xticks([row for row in range(0, len(train_history.history['acc']))])
    plt.title('Train History')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
show_train_history(history)


#####################
# Save model
#####################
try:
    model.save_weights("car_train.h5")
    print("success")
except:
    print("error")

