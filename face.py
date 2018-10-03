# -*- coding=utf-8 -*-
import cv2
import numpy as np
from keras.models import load_model
from utils.preprocessor import preprocess_input
from utils.datasets import get_labels


# 人脸检测器
facePath = "./model/lbpcascade_frontalface.xml"
faceCascade = cv2.CascadeClassifier(facePath)

'''
# 笑脸检测器
smilePath = "/model/haarcascade_smile.xml"
smileCascade = cv2.CascadeClassifier(smilePath)
'''

# 表情
emotionModelPath = './model/emotion_model.hdf5'
emotionClassifier = load_model(emotionModelPath)
emotionLabels = get_labels('fer2013')
# 获取模型的张量
emotionTargetSize = emotionClassifier.input_shape[1:3]


camera = cv2.VideoCapture(0)

while(True):
    ret, frame = camera.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 首先检测人脸，返回的是框住人脸的矩形框
    faces = faceCascade.detectMultiScale(
        grayFrame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # 画出每一个人脸，提取出人脸所在区域
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        grayFace = grayFrame[y:y+h, x:x+w]

        try:
            grayFace = cv2.resize(grayFace, (emotionTargetSize))
        except:
            continue

        # ????????????????????????????????????????????
        grayFace = preprocess_input(grayFace, True)
        grayFace = np.expand_dims(grayFace, 0)
        grayFace = np.expand_dims(grayFace, -1)
        emotionPrediction = emotionClassifier.predict(grayFace)
        emotionProbability = np.max(emotionPrediction)
        emotionLabelArg = np.argmax(emotionPrediction)
        emotionText = emotionLabels[emotionLabelArg]

        bgrFace = frame[y:y+h, x:x+w]

        '''
        # 对人脸进行笑脸检测
        smile = smileCascade.detectMultiScale(
            grayFace,
            scaleFactor=1.16,
            minNeighbors=35,
            minSize=(25, 25),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        '''

        # 框出上扬的嘴角并对笑脸打上Smile标签
        '''
        for (x2, y2, w2, h2) in bgrFace:
            cv2.rectangle(bgrFace, (x2, y2), (x2+w2, y2+h2), (255, 0, 0), 2)
            cv2.putText(frame,'Smile',(x,y-7), 3, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        '''
    cv2.imshow('Emotions', frame)

    if cv2.waitKey(100) & 0xff == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()
