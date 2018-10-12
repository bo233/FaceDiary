import cv2
import numpy as np
from keras.models import load_model
from utils.preprocessor import preprocess_input
import os
import sys
import cv2.face
from GUI.widgets import *
import time


class VideoProcessor:
    RUN_FACE_GENERATE = 0
    RUN_EMOTION_RECOG = 1
    RUN_FACE_RECOG = 2

    runCommand = 1

    def __init__(self, gui):
        self.camera = cv2.VideoCapture(0)
        # 人脸检测器
        self.facePath = './model/lbpcascade_frontalface.xml'
        self.faceCascade = cv2.CascadeClassifier(self.facePath)
        # 表情检测器
        emotionModelPath = './model/emotion_model.hdf5'
        self.emotionClassifier = load_model(emotionModelPath)
        self.emotionLabels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                              4: 'sad', 5: 'surprise', 6: 'neutral'}
        # 记录一个识别表情过程中，各表情出现的次数
        self.emotionRecord = [0, 0, 0, 0, 0, 0, 0]
        # 记录已注册人脸的数量
        self.faceRegistered = 0
        self.whoRU = -1
        # 获取模型的张量
        self.emotionTargetSize = self.emotionClassifier.input_shape[1:3]
        # GUI
        self.gui = gui
        # 人脸识别模型
        self.x = []
        self.y = []
        self.names = []
        self.readImage('./data/at/')
        self.y = np.asarray(self.y, dtype=np.int32)
        self.model = cv2.face.LBPHFaceRecognizer_create()
        # self.model = cv2.face.EigenFaceRecognizer_create()
        self.model.train(np.asarray(self.x), np.asarray(self.y))
        # self.model.save('./model/face_model.xml')
        # self.model = load_model('./model/face_model.xml')

    def faceGenerate(self, name):
        count = 0

        while True:
            ret, frame = self.camera.read()
            try:
                grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(grayFrame, 1.3, 5)

                for (x, y, w, h) in faces:
                    f = cv2.resize(grayFrame[y:y + h, x:x + w], (200, 200))

                    cv2.imwrite('./data/at/%s/%s.pgm' % (str(self.faceRegistered), str(count)), f)
                    count += 1
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, 'Recording you...%d%%' % (count*2), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                self.display(frame)
            except:
                continue
            if cv2.waitKey(85) & 0xff == ord('q'):
                break
            # 每张脸采样50张
            if count > 49:
                self.runCommand = self.RUN_EMOTION_RECOG
                break

        self.x.clear()
        del self.y
        self.y = []
        self.names.clear()
        self.readImage('./data/at/')
        self.y = np.asarray(self.y, dtype=np.int32)
        self.model.train(self.x, self.y)

    def emotionRecog(self, record=False):
        self.emotionRecord = np.zeros(7)
        time_start = time.time()
        time_cost = 0
        while True:
            ret, frame = self.camera.read()
            grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 首先检测人脸，返回的是框住人脸的矩形框
            faces = self.faceCascade.detectMultiScale(
                grayFrame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )

            # 画出每一个人脸，提取出人脸所在区域
            for (x, y, w, h) in faces:
                grayFace = grayFrame[y:y+h, x:x+w]

                try:
                    grayFace = cv2.resize(grayFace, (self.emotionTargetSize))
                except:
                    continue

                grayFace = preprocess_input(grayFace, True)
                grayFace = np.expand_dims(grayFace, 0)
                grayFace = np.expand_dims(grayFace, -1)
                emotionPrediction = self.emotionClassifier.predict(grayFace)
                emotionProbability = np.max(emotionPrediction)
                emotionLabelArg = np.argmax(emotionPrediction)
                emotionText = self.emotionLabels[emotionLabelArg]

                if emotionText == 'angry':
                    color = emotionProbability * np.asarray((255, 0, 0))
                elif emotionText == 'sad':
                    color = emotionProbability * np.asarray((0, 0, 255))
                elif emotionText == 'happy':
                    color = emotionProbability * np.asarray((255, 255, 0))
                elif emotionText == 'surprise':
                    color = emotionProbability * np.asarray((0, 255, 255))
                else:
                    color = emotionProbability * np.asarray((0, 255, 0))

                # 标明心情 框出人脸
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                if record:
                    self.emotionRecord[emotionLabelArg] += 1
                    time_cost = time.time() - time_start
                    cv2.putText(frame, emotionText+' %ds' % (5-time_cost+1), (x, y - 7), 3, 1.2, color, 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, emotionText, (x, y - 7), 3, 1.2, color, 2, cv2.LINE_AA)
            if cv2.waitKey(85) & 0xff == ord('q') | self.runCommand != self.RUN_EMOTION_RECOG:
                break
            if time_cost > 5:
                break
            self.display(frame)
            # if record:


    def readImage(self, path, sz=None):
        self.faceRegistered = 0
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                subjectPath = os.path.join(dirname, subdirname)
                for filename in os.listdir(subjectPath):
                    try:
                        if filename == '.directory':
                            continue
                        elif filename == 'name.txt':
                            filePath = os.path.join(subjectPath, filename)
                            fo = open(filePath, "r+")
                            str = fo.read(10)
                            self.names.append(str)
                            fo.close()
                        elif filename == 'diary.txt':
                            continue
                        else:
                            filePath = os.path.join(subjectPath, filename)
                            im = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)

                            if sz is not None:
                                im = cv2.resize(im, (200, 200))
                            self.x.append(np.asarray(im, dtype=np.uint8))
                            self.y.append(self.faceRegistered)

                    except:
                        print(sys.exc_info()[0])
                        raise
                self.faceRegistered += 1

    def faceRecog(self):
        faceRecord = np.zeros(self.faceRegistered)
        time_start = time.time()
        while True:
            read, img = self.camera.read()
            faces = self.faceCascade.detectMultiScale(img, 1.3, 5)
            for (x, y, w, h) in faces:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                roi = gray[x:x + w, y:y + h]
                try:
                    roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                    params = self.model.predict(roi)
                    faceRecord[params[0]] += 1
                    cv2.putText(img, 'Recognizing you...%ds' % (15-time_cost), (x, y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                except:
                    continue
            self.display(img)
            time_cost = time.time() - time_start
            if cv2.waitKey(85) & 0xff == ord('q') | self.runCommand != self.RUN_FACE_RECOG:
                break
            if time_cost > 15:
                return -1
            if faceRecord.max() > 10:
                return faceRecord.argmax()

    def runControl(self, record=False, name=''):  # record:是否记录人脸或表情出现的次数
        if self.runCommand == self.RUN_EMOTION_RECOG:
            self.emotionRecog(record)
        elif self.runCommand == self.RUN_FACE_GENERATE:
            self.faceGenerate(name)
        elif self.runCommand == self.RUN_FACE_RECOG:
            self.whoRU = self.faceRecog()

    def run(self, command, record=False, name=''):
        self.runCommand = command
        self.runControl(record, name)

    def display(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        self.gui.canvas.add(frame)
        self.gui.window.update_idletasks()
        self.gui.window.update()
