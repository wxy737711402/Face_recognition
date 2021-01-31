'''
本程序用于产生50张灰度图片素材用于后续训练人脸识别系统
'''
import cv2
import os
#路径拼接
face_xml_location=os.path.join('.','haarcascade_frontalface_default.xml')
#级联分级器
detector = cv2.CascadeClassifier(face_xml_location)
#opencv函数调用摄像头，默认为摄像头0
camera = cv2.VideoCapture(0)
#设置素材照片第一个为0
pic_num = len(os.listdir('.\\face_pic'))
cycl_number=pic_num+20
ID = input('enter your id: ')
while True:
    ret, img = camera.read() #摄像头读取照片
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #色彩空间转换
    #opencv检测人脸函数(灰度图，搜索窗口的比例系数，构成检测目标的相邻矩形的最小个数)
    faces = detector.detectMultiScale(gray,1.3, 5)
    for (x, y, w, h) in faces:
        #rectangle(矩阵的左上点坐标,矩阵的右下点坐标,画线对应的rgb颜色,所画的线的宽度)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        pic_num = pic_num + 1
        pic_save_location=os.path.join('.','face_pic','user%s.%s.jpg')
        cv2.imwrite(pic_save_location%(ID,pic_num), gray[y:y + h, x:x + w])
        cv2.imshow('frame', img)
    # 设置相片取样间隔
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # 设置采样相片数
    elif pic_num >= cycl_number:
        break

camera.release()
cv2.destroyAllWindows()
