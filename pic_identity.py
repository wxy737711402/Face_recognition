'''
在一张图片上识别出人脸，并在脸周围绘制一个矩形，最后把绘制矩形的图片保存到一张新图片
'''

import cv2
import os

#os.path.join函数会根据系统返回一个路径字符串
filename=os.path.join('.','E:/test1.jpg')
savefilename=os.path.join('.','test2.jpg')

def detect(filename):
    face_xml_location=os.path.join('.','haarcascade_frontalface_default.xml')
    #声明cascadeclassifie对象face_cascade，用于检测人脸
    face_cascade=cv2.CascadeClassifier(face_xml_location)

    img=cv2.imread(filename)
    #openCV识别灰度图片，也就是黑白图片，所以要进行转换
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    
    #在人脸周围绘制矩形
    for (x,y,w,h) in faces:
        img=cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),0)
        cv2.namedWindow('liuyifeidetected')
        cv2.imshow('liuyifeidetected',img)
        cv2.imwrite(savefilename,img)
        cv2.waitKey(0)

#图片人脸检测程序，识别图片pic.jpg
detect(filename)
