#encoding:utf-8

'''
author: zzm
usage: 
1. 获取人脸图片
   python2 face.py getface faces 
2. 人脸识别
   python2 face.py recognize faces 
'''

import sys
import cv2
import os
import numpy as np
from facerec.model import PredictableModel
from facerec.feature import Fisherfaces
from facerec.distance import EuclideanDistance
from facerec.classifier import NearestNeighbor
from detector import CascadedDetector

image_size = (100,100)   #统一人脸图片大小
capture = cv2.VideoCapture(0)   #获取摄像头
action = sys.argv[1]    #从命令行读取要进行的操作
dataset = sys.argv[2]   #从命令行读取人脸图片集合

#人脸检测器
detector = CascadedDetector(cascade_fn="haarcascade_frontalface_alt2.xml", minNeighbors=5, scaleFactor=1.1)

#在摄像头图像中标记人名
def draw_str(dst, (x, y), s):
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

#获取模型
def get_model():
    c = 0
    X = []   #人脸图片特征向量
    y = []   #每张图片标签,和人名一一对应
    name = dict()
    for dirname, dirnames, filenames in os.walk(dataset):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                #读取图片并调整大小
                im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                im = cv2.resize(im, image_size)
                X.append(np.asarray(im, dtype=np.uint8))
                y.append(c)
            name[c] = subdirname
            c += 1
    #获取模型
    feature = Fisherfaces()
    classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
    model = PredictableModel(feature=feature, classifier=classifier)
    #训练模型
    model.compute(X,y)
    return model, name

#获取人脸图片
def get_face():
    #建立图片目录
    person = sys.argv[3]
    dirname = dataset + '/' + person
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    c = 1
    while True:
        # 从摄像头获取一帧图像
        ret, frame = capture.read()
        img = cv2.resize(frame, (frame.shape[1]/2, frame.shape[0]/2), interpolation = cv2.INTER_CUBIC)
        # 检测出的第一个人脸矩形
        try:
            rect = detector.detect(img)[0]
            # 获取矩形框内的人脸图片
            x0,y0,x1,y1 = rect
            face = img[y0:y1, x0:x1]
            #转灰度图
            face = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, (92,112), interpolation = cv2.INTER_CUBIC)
        except:
            pass
        cv2.imshow('getface', frame)
        #获取键盘输入
        ch = cv2.waitKey(1)
        # 按esc键退出
        if ch == 27:
            break
        # 空格键截图
        if ch == ord(' '):
            cv2.imwrite(dirname+'/'+str(c)+".bmp", face)
            print(str(c)+".bmp")
            c+=1


#人脸识别
def recognize():
    model, name = get_model()
    print(name)
    names = ''
    dic = {}
    while True:
        # 从摄像头获取一帧图像
        ret, frame = capture.read()
        img = cv2.resize(frame, (int(frame.shape[1]/1.5), int(frame.shape[0]/1.5)), interpolation = cv2.INTER_CUBIC)
        imgout = img.copy()
        # 对于每一个检测出的人脸矩形
        for rect in detector.detect(img):
            # 获取矩形框内的人脸图片
            x0,y0,x1,y1 = rect
            face = img[y0:y1, x0:x1]
            #转灰度图
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = cv2.resize(face, image_size, interpolation = cv2.INTER_CUBIC)
            prediction = model.predict(face)[0]
            #标出脸部区域
            cv2.rectangle(imgout, (x0,y0),(x1,y1),(0,255,0),2)
            #画出人名
            draw_str(imgout, (x0-20,y0-20), name[prediction])
            if name[prediction] not in dic:
                dic[name[prediction]] = 1;
                names += name[prediction]+'\n'
            #prediction[max],count,
        cv2.imshow('recgonize', imgout)
        ch = cv2.waitKey(1)
        # 按esc键退出
        if ch == 27:
            break
    with open('name.txt','w') as f:
        f.write(names)

#获取人脸图片 或者 人脸识别
get_face() if action=='getface' else recognize()