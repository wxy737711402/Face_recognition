import cv2
import numpy as np
import os
#训练器创建
recognizer = cv2.face.LBPHFaceRecognizer_create()
#拼接
train_data_location=os.path.join('.','train_data','train.yml')
#读取地址
recognizer.read(train_data_location)

#文件拼接
face_xml_location=os.path.join('.','haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(face_xml_location)
camera = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, im = camera.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)#转为灰度图像
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w -50, y+h-50), (0, 0, 255), 2)
        img_id,conf = recognizer.predict(gray[y : y + h , x:x + w])
        if (conf > 25):
            print(img_id)
            if(img_id <= 100):
                name = 'wxy'
        else:
            name = "Unknown"
    
        cv2.putText(im, str(name), (x, y + h), font, 0.55, (0, 255, 0), 1)
    cv2.imshow('im', im)
    cv2.imshow('gray',gray)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
