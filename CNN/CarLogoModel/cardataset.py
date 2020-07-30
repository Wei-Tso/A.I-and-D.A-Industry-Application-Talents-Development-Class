import cv2
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

def load_data():
    #####################
    # 載入 csv 檔，並讀取所有資料
    #####################
    with open('./carlogo.csv', 'r') as file:
        rows = file.readlines()
    
    x = []
    y = []
    label = 0
    #####################
    # 讀取每一行資料，從 1 開始 rows[1:]
    # 資料會顯示為：'car1,1,0,0,0,0,0,0,0,0,0,0,0\n', 'car2,1,0,0,0,0,0,0,0,0,0,0,0\n', ...
    # rows[0] 為 column_name：image_id,benz,bmw,ford,honda,lansus,luxgan,mazda,mitsubishi,nissan,suzuki,toyota,volkswagen
    #####################
    for row in rows[1:]:
        row = row.replace('\n','')

        #####################
        # filename：圖片名稱
        #####################
        filename = row.split(",")[0]
        benz = row.split(",")[1]
        bmw = row.split(",")[2]
        ford = row.split(",")[3]
        honda = row.split(",")[4]
        lansus = row.split(",")[5]
        luxgen = row.split(",")[6]
        mazda = row.split(",")[7]
        mitsubiish = row.split(",")[8]
        nissan = row.split(",")[9]
        suzuki = row.split(",")[10]
        toyota = row.split(",")[11]
        volkswagen = row.split(",")[12]

        #####################
        # 利用 filename 把資料集(carlogomap)的所有圖片 load 進來
        #####################
        img = cv2.imread("./carlogomap/" + filename + ".jpg" , cv2.IMREAD_COLOR)
        img = cv2.resize(img , (64 , 64))
        x.append(img)

        #####################
        # 利用 .split(",")[] 的結果，對每一張圖片進行 label
        #####################
        if benz == "1":
            label = 0
        elif bmw == "1":
            label = 1
        elif ford == "1":
            label = 2
        elif honda =="1":
            label = 3
        elif lansus =="1":
            label = 4
        elif luxgen =="1":
            label = 5
        elif mazda =="1":
            label = 6
        elif mitsubiish =="1":
            label = 7
        elif nissan =="1":
            label = 8
        elif suzuki =="1":
            label = 9
        elif toyota =="1":
            label = 10
        elif volkswagen =="1":
            label = 11
        
        y.append(label)

    x = np.array(x)
    y = np.array(y)
    
    np.save("train_x.npy" , x)
    np.save("train_y.npy" , y)
    
    return x , y

times = 3

def load_data_augmentation():
    with open('./carlogo.csv' , 'r') as file:
        rows = file.readlines()
    
    x = []
    y = []
    label = 0
    for row in rows[1:]:
        row = row.replace('\n','')

        filename = row.split(",")[0]
        benz = row.split(",")[1]
        bmw = row.split(",")[2]
        ford = row.split(",")[3]
        honda = row.split(",")[4]
        lansus = row.split(",")[5]
        luxgen = row.split(",")[6]
        mazda = row.split(",")[7]
        mitsubishi = row.split(",")[8]
        nissan = row.split(",")[9]
        suzuki = row.split(",")[10]
        toyota = row.split(",")[11]
        volkswagen = row.split(",")[12]
        
        img = cv2.imread("./carlogomap/" + filename + ".jpg" , cv2.IMREAD_COLOR)
        img = cv2.resize(img , (64 , 64))
        
        x.append(img)   

        if benz == "1":
            label = 0
        elif bmw == "1":
            label = 1
        elif ford == "1":
            label = 2
        elif honda =="1":
            label = 3
        elif lansus =="1":
            label = 4
        elif luxgen =="1":
            label = 5
        elif mazda =="1":
            label = 6
        elif mitsubishi =="1":
            label = 7
        elif nissan =="1":
            label = 8
        elif suzuki =="1":
            label = 9
        elif toyota =="1":
            label = 10
        elif volkswagen =="1":
            label = 11  

        y.append(label)
        
        
        #####################
        # 對資料集進行 brightness augmentation
        #####################
        # brightness augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data , 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(brightness_range = [0.2 , 1.0])
        # prepare iterator
        it = datagen.flow(samples , batch_size = 1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)
        
        #####################
        # 對資料集進行 horizontal shift augmentation
        #####################
        '''
        # horizontal shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data , 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(width_shift_range = [-200 , 200])
        # prepare iterator
        it = datagen.flow(samples , batch_size=1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label) 
        '''       
        
        #####################
        # 對資料集進行 vertical shift augmentation
        #####################
        '''
        # vertical shift augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data , 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(height_shift_range = [-200 , 200])
        # prepare iterator
        it = datagen.flow(samples , batch_size = 1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label) 
        '''

        #####################
        # 對資料集進行 rotation augmentation
        #####################
        '''
        # rotation augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data , 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(rotation_range=80)
        # prepare iterator
        it = datagen.flow(samples , batch_size = 1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label)  
        '''

        #####################
        # 對資料集進行 zoom augmentation
        #####################
        # zoom augmentation
        data = img_to_array(img)
        # expand dimension to one sample
        samples = expand_dims(data , 0)
        # create image data augmentation generator
        datagen = ImageDataGenerator(zoom_range = [0.5,1.0])
        # prepare iterator
        it = datagen.flow(samples , batch_size = 1)
        # generate samples and plot
        for i in range(times):
            # generate batch of images
            batch = it.next()
            # convert to unsigned integers for viewing
            image = batch[0].astype('uint8')
            x.append(image)
            y.append(label) 

        #####################
        # 對資料集進行 鏡像 augmentation
        #####################
        # extra augmentation
        if bmw == "1":
            img_hr = cv2.flip(img , 1 , dst = None) #水平鏡像
            x.append(img_hr)
            y.append(1)
        
            img_vr = cv2.flip(img , 0 , dst = None) #垂直鏡像
            x.append(img_vr)
            y.append(1)
            
            '''
            img_sr = cv2.flip(img , -1 , dst=None) #對角鏡像
            x.append(img_sr)
            y.append(1)
            '''
            '''
            # rotation augmentation
            data = img_to_array(img)
            # expand dimension to one sample
            samples = expand_dims(data , 0)
            # create image data augmentation generator
            datagen = ImageDataGenerator(rotation_range = 270)
            # prepare iterator
            it = datagen.flow(samples , batch_size=1)
            # generate samples and plot
            for i in range(10):
                # generate batch of images
                batch = it.next()
                # convert to unsigned integers for viewing
                image = batch[0].astype('uint8')
                x.append(image)
                y.append(1)  
            '''

    x = np.array(x , dtype = np.float32)
    y = np.array(y)
    
    np.save("train_x_aug.npy" , x)
    np.save("train_y_aug.npy" , y)
    
    return x , y



# load_data_augmentation()