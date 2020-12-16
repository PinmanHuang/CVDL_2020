"""
Author  : Joyce
Date    : 2020-12-15
"""
from UI.hw2_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

class PyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        # === push button clicked action === 
        self.pushButton.clicked.connect(self.bg_sub)
        self.pushButton_2.clicked.connect(self.preprocessing)
        self.pushButton_3.clicked.connect(self.tracking)

    def bg_sub(self):
        opencv = OpenCv()
        opencv.Q1_1()

    def preprocessing(self):
        opencv = OpenCv()
        opencv.Q2_1()

    def tracking(self):
        opencv = OpenCv()
        opencv.Q2_2()

class OpenCv(object):
    def __init__(self):
        super(OpenCv, self).__init__()

    def Q1_1(self):
        i = mean = std = 0
        frames = []
        cv2.namedWindow('1.1 Original Video', cv2.WINDOW_NORMAL)
        cv2.namedWindow('1.1 Subtraction Video', cv2.WINDOW_NORMAL)
        # capture video and get fps
        cap = cv2.VideoCapture('./Q1_Image/bgSub.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        # print("fps: {}".format(fps))
        while(cap.isOpened()):
            i = i + 1
            # get frames of video and convert to gray
            # frame shape: (176, 320), type: ndarray
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = np.zeros_like(gray)

            # process according to frame
            if i < 50:
                # 1st to 49th frame
                frames.append(gray)
            elif i == 50:
                # 50th frame
                frames.append(gray)
                all_frames = np.array(frames)
                # mean and standard deviation
                mean = np.mean(all_frames, axis=0)
                std = np.std(all_frames, axis=0)
                # print("type: {}, shape: {}, mean: {}, std: {}".format(
                #      type(all_frames), all_frames.shape, mean.shape, std.shape))
                # if standard deviation is less then 5, set to 5
                std[std < 5] = 5
            else:
                # after 51th frame
                # subtract
                diff = np.subtract(gray, mean)
                diff = np.absolute(diff)
                result[diff > 5*std] = 255
                # print("type: {}, shape: {}, diff_type:{}, diff_shape: {}, mean: {}, std: {}".format(
                #      type(gray), gray.shape, type(diff), diff.shape, mean.shape, std.shape))
                     
            cv2.imshow('1.1 Original Video', frame)
            cv2.imshow('1.1 Subtraction Video', result)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def Q2_1(self):
        cv2.namedWindow('2.1 Preprocessing Video', cv2.WINDOW_NORMAL)
        # Set up the detector with parameters
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 10
        params.maxThreshold = 200
        params.filterByArea = True
        params.minArea = 35
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 0.9
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)

        # capture video
        cap = cv2.VideoCapture('./Q2_Image/opticalFlow.mp4')
        # get 1st frame of video
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect blobs
        keypoints = detector.detect(gray)

        # Draw rectangle and line
        for kp in keypoints:
            x, y = (kp.pt)
            x = int(x)
            y = int(y)
            frame = cv2.rectangle(frame, (x-5, y-5), (x+5, y+5), (0, 0, 255), 1)
            frame = cv2.line(frame, (x, y-5), (x, y+5), (0, 0, 255), 1)
            frame = cv2.line(frame, (x-5, y), (x+5, y), (0, 0, 255), 1)

        cv2.imshow('2.1 Preprocessing Video', frame)
        cv2.waitKey(0)
        cap.release()
        cv2.destroyAllWindows()

    def Q2_2(self):
        cv2.namedWindow('2.2 Video tracking', cv2.WINDOW_NORMAL)
        lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        # Create some random colors
        color = np.random.randint(0,255,(100,3))
        # Set up the detector with parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 35
        params.filterByCircularity = True
        params.minCircularity = 0.8
        params.maxCircularity = 0.9
        params.filterByConvexity = True
        params.minConvexity = 0.5
        params.filterByInertia = True
        params.minInertiaRatio = 0.5
        detector = cv2.SimpleBlobDetector_create(params)

        # capture video and get fps
        cap = cv2.VideoCapture('./Q2_Image/opticalFlow.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        # get 1st frame of video
        ret, frame = cap.read()
        gray_1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect blobs
        keypoints = detector.detect(gray_1)
        p0 = np.array([[[kp.pt[0], kp.pt[1]]] for kp in keypoints]).astype(np.float32)
        mask = np.zeros_like(frame)

        while(cap.isOpened()):
            # get frames of video and convert to gray
            # frame shape: (176, 320), type: ndarray
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            p1, st, err = cv2.calcOpticalFlowPyrLK(gray_1, gray, p0, None, **lk_params)
            # Select good points
            good_new = p1[st==1]
            good_old = p0[st==1]

            # draw the tracks
            for i,(new,old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (a, b), (c, d), (0, 0, 255), 2)
                frame = cv2.circle(frame, (a, b), 3, (0, 0, 255), -1)
            result = cv2.add(frame, mask)
            

            cv2.imshow('2.2 Video tracking', result)
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break

            gray_1 = gray.copy()
            p0 = good_new.reshape(-1, 1, 2)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())