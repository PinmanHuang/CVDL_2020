"""
Author  : Joyce
Date    : 2020-10-12
"""
from UI.hw1_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import cv2
import numpy as np

class PyMainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        # === push button clicked action === 
        self.pushButton.clicked.connect(self.find_corner)
        self.pushButton_2.clicked.connect(self.find_intrinsic)
        self.pushButton_3.clicked.connect(self.find_distortion)
        self.pushButton_4.clicked.connect(self.find_extrinsic)
        # === combo box change action ===
        self.comboBox.currentIndexChanged.connect(self.select_image)
        self.show()

        # === Q1 ===
        self.opencv = OpenCv()
        self.opencv.Q1()
        self.q1_3_img_idx = 0

    """
    ref: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    """ 
    def find_corner(self):
        print('find corner')
        self.opencv.Q1_1()
        # # termination criteria
        # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        # objp = np.zeros((8*11, 3), np.float32)
        # objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # # Arrays to store object points and image point from all the image
        # objpoints = []  # 3d point in real world space
        # imgpoints = []  # 2d points in image plane
        
        # for f_idx in range(1, 16):
        #     # Read image and converty to gray image
        #     img = cv2.imread('./Q1_Image/'+str(f_idx)+'.bmp')
        #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        #     # Find the chess board corners
        #     ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            
        #     # If found, add object points, image points
        #     if ret == True:
        #         objpoints.append(objp)

        #         corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        #         imgpoints.append(corners)

        #         # Draw and display the corner
        #         cv2.drawChessboardCorners(img, (8, 11), corners, ret)

        #         cv2.namedWindow('1.1 Find Corners: '+str(f_idx)+'.bmp', cv2.WINDOW_NORMAL)
        #         cv2.imshow('1.1 Find Corners: '+str(f_idx)+'.bmp', img)
        #         # cv2.waitKey(0)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    def find_intrinsic(self):
        print('find intrinsic')
        self.opencv.Q1_2()

    def find_distortion(self):
        print('find distortion')
        self.opencv.Q1_4()
    
    def find_extrinsic(self):
        print('find extrinsic')
        self.opencv.Q1_3(self.q1_3_img_idx)

    def select_image(self, text):
        print('select image: {}'.format(str(int(text)+1)))
        self.q1_3_img_idx = int(text)
        

class OpenCv(object):
    def __init__(self):
        super(OpenCv, self).__init__()
        # === Q1 ===
        self.corner_img = np.empty(15, dtype=object)    # corner images
        # Arrays to store object points and image point from all the image
        self.objpoints = []                             # 3d point in real world space
        self.imgpoints = []                             # 2d points in image plane
        self.gray_img = np.empty(15, dtype=object)          # gray images
        # ==========

    def Q1(self):
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # # Arrays to store object points and image point from all the image
        # objpoints = []  # 3d point in real world space
        # imgpoints = []  # 2d points in image plane
        
        for f_idx in range(1, 16):
            print(f_idx)
            # Read image and converty to gray image
            img = cv2.imread('./Q1_Image/'+str(f_idx)+'.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.gray_img[f_idx-1] = gray
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.imgpoints.append(corners)

                # Draw and display the corner
                cv2.drawChessboardCorners(img, (8, 11), corners, ret)

                # save corner images
                self.corner_img[f_idx-1] = img
            

    def Q1_1(self):
        print('Q1_1')
        for f_idx in range(1, 16):
            print(f_idx)
            cv2.namedWindow('1.1 Find Corners: '+str(f_idx)+'.bmp', cv2.WINDOW_NORMAL)
            cv2.imshow('1.1 Find Corners: '+str(f_idx)+'.bmp', self.corner_img[f_idx-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Q1_2(self):
        print('Q1_2: 15.bmp')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[14].shape[::-1], None, None)
        print(mtx)

    def Q1_3(self, img_idx):
        print('Q1_3: '+str(img_idx+1)+'.bmp')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[img_idx].shape[::-1], None, None)
        print(cv2.Rodrigues(rvecs[img_idx])[0])
        print(tvecs[img_idx])

    def Q1_4(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[14].shape[::-1], None, None)
        print(dist)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())