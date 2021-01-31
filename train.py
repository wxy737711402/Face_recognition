import cv2
import os
import numpy as np
from PIL import Image
#拼接
face_xml_location=os.path.join('.','haarcascade_frontalface_default.xml')
#级联分级器
detector = cv2.CascadeClassifier(face_xml_location)
#创建LBPH识别器开始训练
recognizer = cv2.face.LBPHFaceRecognizer_create()


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = [] #面部样本
    ids = []          #标识id

    for image_path in image_paths:
        image = Image.open(image_path).convert('L') #转换为灰度图像
        image_np = np.array(image, 'uint8')#将数据转化为矩阵 无符号整形
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':
            continue
        image_id = int(os.path.split(image_path)[-1].split(".")[1])
        faces = detector.detectMultiScale(image_np) #数据矩阵
        for (x, y, w, h) in faces:
            face_samples.append(image_np[y:y + h, x:x + w]) #添加信息至数组
            ids.append(image_id)#添加信息至数组
    print("训练完成")

    return face_samples, ids
#路径拼接
tain_data_location=os.path.join('.','train_data','train.yml')
face_data_location=os.path.join('.','face_pic')

faces, Ids = get_images_and_labels(face_data_location)
recognizer.train(faces, np.array(Ids)) #训练器训练
recognizer.save(tain_data_location)    #训练结果保存
