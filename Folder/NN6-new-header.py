# -*- coding: utf-8 -*-
"""
NN6-new.py
Created on Sat Nov 21 09:40:54 2020

@author: pchw8598
"""
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  

import matplotlib.pyplot as plt   
from os import listdir
import glob
import cv2
import numpy as np
import os
import util6 as u

#============= obtain trainging images' path =================
training_img_path='..\\pic_numbers_ok\\training\\'
print('Images\' path= '+training_img_path)

training_img_dirs=listdir(training_img_path)  # obtain all directories of path 'img_path'

print('All ',len(training_img_dirs),' directories. ') #show  number of directories
print(training_img_dirs) #show all directories

#============= 擷取路徑中的檔名 ===============================
# extract the filename from a path set
def extractFileName(files):    
    for item in files:
        filename=os.path.basename(item) #exreact the filename from a path
        print(filename,end=' ')

#============= load customized images =========================
# path: the path containing directories of images
def load_customized_data(path, fgFileShow=False, pre_adjust=False):   
    imgs, labs,all_files = read_images(path, fgFileShow)
    
    if len(imgs)!=200: # the number of images is incorrect
        print('\033[1;31m',end='')
    else:
        print('\033[1;37m',end='')
                        
    print('影像的shape=',imgs.shape,', 標籤的shape=',labs.shape)  
    
    print('\033[0m',end='')  #恢復正常顏色
    
    return imgs, labs,all_files 

#============== 從資料夾載入影像 ===============================
# 由 path 載入圖檔, 並以子資料夾名為標籤 (0~9的資料夾)
def read_images(path, fgFileShow=False, pre_adjust=False):
    test_images=[]  # 建立空的list，用於儲存影像
    test_labels=[]  # 建立空的list  ，用於儲存標籤  
    all_files=[]  #建立空的list，用於儲存影像檔名
                    
    for label in range(10):  #數字0-9
        #files：某個數字資料夾裡的所有檔案
        files = glob.glob(path + "/" + str(label) + "/*.png" ) 
        all_files.extend(files) #將影像檔名加入all_files串列
        
        if fgFileShow==True:
            extractFileName(files) #顯示所有的檔案名稱
            
        for file in files:
            img = cv2.imread(file)  #讀取影像檔案
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #轉灰階
            img = cv2.bitwise_not(img)       # 反白：變成黑底白字
            img = cv2.resize(img, (28, 28))  # 重設大小為 28x28
            
            if pre_adjust:
                img = u.img_pre_adjust(img)  # 影像前置調

            test_images.append(img)   #放入調整後的影像
            test_labels.append(label) #放入標籤

    return (np.array(test_images), np.array(test_labels), all_files)

#============= show images ===========================
#start:圖的開始編號 num:要顯示多少張圖
def drawImages(imgs,labs,all_files,start,num):    
    columnNum=10 #每列顯示多少張影像
    rowNum=int(num/columnNum) #需要顯示多少列的影像    
    imgs_num=len(imgs)
    
    plt.gcf().set_size_inches(15,4)
    for i in range(num):        
        no=i+start;        

        if no >=imgs_num: #the image's number is incorrect
            print('\033[1;31m ERROR!! \033[0m')
            break;
        
        ax=plt.subplot(rowNum,columnNum,1+i)
        ax.imshow(imgs[no],cmap='binary')
        
        title = 'label = ' + str(labs[no])+'\n'+\
                 os.path.basename(all_files[no])       
        ax.set_title(title,fontsize=12)
    
        ax.set_xticks([]) #不顯示X的刻度
        ax.set_yticks([]) #不顯示y的刻度    

    plt.show()  
    
#============= 驗證是否正確 ===========================
#start:圖的開始編號 num:要顯示多少張圖
def drawPredict(imgs,labs,predict,all_files, start,num):    
    columnNum=10 #每列顯示多少張影像
    rowNum=int(num/columnNum) #需要顯示多少列的影像    
    
    plt.gcf().set_size_inches(15,4)
    for i in range(num):
        no=i+start;
        ax=plt.subplot(rowNum,columnNum,1+i)
        ax.imshow(imgs[no],cmap='binary')
        
        title = 'label = ' + str(labs[no])
        if labs[no] == predict[no]:
            title += '\npredi = ' + str(predict[no])
        else:
            title += '\npre● = ' + str(predict[no]) 
            
        title+='\n'+os.path.basename(all_files[no])    
        ax.set_title(title,fontsize=12)
    
        ax.set_xticks([]) #不顯示X的刻度
        ax.set_yticks([]) #不顯示y的刻度    

    plt.show()   
    
#============ 資料前處理 ====================
def data_pre_proc(imgs, labs):    
    if imgs != np.ndarray:     #如果imgs不是ndarray的型別，
        imgs = np.array(imgs)  #則轉為numpy 的array形別       
    
    x = imgs.reshape((len(imgs), 784))
        
    x= x.astype('float32') / 255
    y  = to_categorical(labs)
    
    return x, y

#===========   ================================
def process_all(img_path,img_dirs,adj_size=0,fgImgShow=False,fgFileShow=False):
    imgs=[] #所有的影像
    labs=[]  #所有的標籤
    files=[] #所有的檔案
    
    for index in range(0,len(img_dirs)): 
        path=img_path+img_dirs[index];
        print('no.',index+1,'：',path,'：',end='')
        imgset, labset,fileset=load_customized_data(path,fgFileShow,
                                                    pre_adjust=True)   
        #--------------- 調整影像 -------------------------------------
        if adj_size > 0:                        
            for i in range(len(imgset)):             
                imgset[i] = u.img_best(imgset[i], size=adj_size, 
                                       vdif=1, hdif=1)
       
        # ---- show all images ----
        if fgImgShow==True: 
            for i in range(0,len(imgset),20):
                drawImages(imgset, labset,fileset,i,20 )
            
        imgs.extend(imgset) 
        labs.extend(labset)
        files.extend(fileset)
    
    return imgs,labs, files

#============= 訓練樣本  ======================
fgShowImage=True
fgShowFile=False
adj_size=19
fns='_'+str(adj_size)

train_images,train_labs,train_files=process_all(training_img_path,training_img_dirs,adj_size,fgShowImage,fgShowFile)

#=========預處理訓練資料==========================
print('訓練樣本數量',len(train_images),end=' ')
print('訓練標籤數量',len(train_labs),end=' ')
print('訓練檔案數量',len(train_files))

x_train, y_train=data_pre_proc(train_images,train_labs)
print(x_train.shape)
print(y_train.shape)

#---------開啟建立 NN 模型-------------
model=Sequential()
model.add(Dense(512,activation='relu',input_dim=784))
model.add(Dense(10,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['acc'])


#檢視模型的摘要
print(model.summary())

#==========訓練模型================
history=model.fit(x_train,y_train,epochs=5,batch_size=120)

#==========儲存模型================
#model.save('customized.h5')

#--------刪除 NN model -------------
#del model

#========重新載入模型===============
from tensorflow.keras.models import load_model
#model=load_model('customized.h5')

#============測試樣本================
test_img_path='..\\pic_numbers_ok\\test\\'
print('Images\' path= '+test_img_path)

test_img_dirs=listdir(test_img_path)

print('All ',len(test_img_dirs),' directories. ')

print(test_img_dirs)

#===========載入測試影像、建立標籤============
fgShowImage=True
fgShowFile=False

test_images, test_labs, test_files=process_all(test_img_path,test_img_dirs,adj_size,fgShowImage,fgShowFile)

#==========預處理測試資料===============
print('測試樣本數量',len(test_images),end=' ')
print('測試標籤數量',len(test_labs),end=' ')
print('測試檔案數量',len(test_files))

x_test, y_test=data_pre_proc(test_images,test_labs)
print(x_test.shape)
print(y_test.shape)


#===========評估模型=================
test_loss,test_acc=model.evaluate(x_test,y_test,verbose=0)
print('測試資料的遺失率=',test_loss)
print('測試資料的準確率=',test_acc)

#===========預測=====================
#predict=model.predict(x_test)

predict=model.predict_classes(x_test)
print("digitals=",predict)

#===========顯示影像，驗證是否正確=======
for i in range(0,len(test_images),20):
    drawPredict(test_images, test_labs, predict,test_files,i,20 )



















