# coding= utf-8
from read_data import read_name_list,read_file
from train_model import Model
import cv2
import os 
import numpy as np
from read_img import endwith
from dataSet import DataSet
from keras import backend as K
from keras.utils import np_utils


K.clear_session()
# def test_onePicture(path):
#     model= Model()
#     model.load()
#     img = cv2.imread(path)
#     picType,prob = model.predict(img)
#     if picType != -1:
#         name_list = read_name_list('F:\myProject\pictures\dataset')
#         print(name_list)
#         print( name_list[picType],prob)
#     else:
#         print (" Don't know this person")

# #读取文件夹下子文件夹中所有图片进行识别
# def test_onBatch(path):
#     model= Model()
#     model.load()
#     index = 0
#     img_list, label_list, counter = read_file(path)
#     for i in range(len(img_list)):
#         picType,prob = model.predict(img_list[i])
#         if picType==label_list[i] & picType != -1:
#             index += 1
#     #计算预测正确的概率
#     pro_predict=float(index)/len(img_list)
#     return pro_predict

#读取文件夹下子文件夹中所有图片进行识别
def test_onBatch(path):
    model= Model()
    model.load()
    index = 0
    img_list, label_list, counter = read_file(path)
#     img_list = img_list.reshape(img_list.shape[0], 174, 212, 1)
#     print(img_list.shape[0:])
#     img_list = img_list.astype('float32')/255
#     Label_list = np_utils.to_categorical(label_list, num_classes=counter)
    for img in img_list:
        picType,prob = model.predict(img)
        if picType != -1:
            index += 1
            name_list = read_name_list('G:/desktop/myProject/pictures/test')
            print(name_list)
            print (name_list[picType])
        else:
            print (" Don't know this person")
  
    return index




# # 新建一个读取测试数据的函数  由于数据集的顺序是从1开始的 所以dir_counter是从1开始的
# def read_testfile(path):
#     img_list=[]
#     label_lsit=[]
#     dir_counter=1
#     # 将不同文件夹中的测试集读取  因为数据集中的数据已经长宽一致所有没有resize这个操作
#     for child_dir in os.listdir(path):
#         child_path=os.path.join(path,child_dir)

#         for dir_image in os.listdir(child_path):
#             print(child_path) 
#             if endwith(dir_image,'jpg'):
#                 img=cv2.imread(os.path.join(child_path,dir_image))
#                 img_list.append(img)
#                 label_lsit.append(dir_counter)

    #     dir_counter += 1
    # # 返回的img_list转成 np.array的格式
    # img_list =np.array(img_list)
    # return img_list,label_lsit,dir_counter

if __name__ == '__main__':
    # test_onePicture('F:\myProject\pictures\pic4.jpg')
    pro_predict=test_onBatch('G:/desktop/myProject/pictures/test/')
    #print("Prediction of Accuracy: %f",pro_predict)



