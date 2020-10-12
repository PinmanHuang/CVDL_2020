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

        
        
    def find_corner(self):
        print('find corner')
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:8, 0:11].T.reshape(-1, 2)

        # Arrays to store object points and image point from all the image
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane
        
        for f_idx in range(1, 16):
            # Read image and converty to gray image
            img = cv2.imread('./Q1_Image/'+str(f_idx)+'.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 11), None)
            
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)

                # Draw and display the corner
                cv2.drawChessboardCorners(img, (8, 11), corners, ret)

                cv2.namedWindow('1.1 Find Corners: '+str(f_idx)+'.bmp', cv2.WINDOW_NORMAL)
                cv2.imshow('1.1 Find Corners: '+str(f_idx)+'.bmp', img)
                # cv2.waitKey(0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def find_intrinsic(self):
        print('find intrinsic')

    def find_distortion(self):
        print('find distortion')
    
    def find_extrinsic(self):
        print('find extrinsic')

    def select_image(self, text):
        print('select image: {}'.format(str(int(text)+1)))
        

class OpenCv(object):
    def __init__(self):
        super(OpenCv, self).__init__()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())