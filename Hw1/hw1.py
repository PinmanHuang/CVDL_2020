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
from matplotlib import pyplot as plt

class PyMainWindow(QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        # === push button clicked action === 
        self.pushButton.clicked.connect(self.find_corner)
        self.pushButton_2.clicked.connect(self.find_intrinsic)
        self.pushButton_3.clicked.connect(self.find_distortion)
        self.pushButton_4.clicked.connect(self.find_extrinsic)
        self.pushButton_5.clicked.connect(self.augmented_reality)
        self.pushButton_6.clicked.connect(self.stereo_disparity)
        self.pushButton_8.clicked.connect(self.keypoints)
        self.pushButton_9.clicked.connect(self.matched_keypoints)
        self.pushButton_10.clicked.connect(self.show_train_images)
        self.pushButton_11.clicked.connect(self.show_hyperparameters)
        self.pushButton_12.clicked.connect(self.show_model_strucuture)
        self.pushButton_13.clicked.connect(self.show_accuracy)
        self.pushButton_14.clicked.connect(self.test)

        # === combo box change action ===
        self.comboBox.currentIndexChanged.connect(self.select_image)
        self.show()

        # === Q1 ===
        self.q1_3_img_idx = 0

    """
    ref: https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    """ 
    def find_corner(self):
        print('Find corner')
        opencv = OpenCv()
        opencv.Q1()
        opencv.Q1_1()

    def find_intrinsic(self):
        print('Find intrinsic')
        opencv = OpenCv()
        opencv.Q1()
        opencv.Q1_2()

    def find_distortion(self):
        print('Find distortion')
        opencv = OpenCv()
        opencv.Q1()
        opencv.Q1_4()
    
    def find_extrinsic(self):
        print('Find extrinsic')
        opencv = OpenCv()
        opencv.Q1()
        opencv.Q1_3(self.q1_3_img_idx)

    def select_image(self, text):
        self.q1_3_img_idx = int(text)

    def augmented_reality(self):
        print('Augmented reality')
        opencv = OpenCv()
        opencv.Q2()

    def stereo_disparity(self):
        print('Stereo disparity')
        opencv = OpenCv()
        opencv.Q3()

    def keypoints(self):
        print('Ketpoints')
        opencv = OpenCv()
        opencv.Q4_1()

    def matched_keypoints(self):
        print('Matched keypoints')
        opencv = OpenCv()
        opencv.Q4_2()

    def show_train_images(self):
        print('Show train images')
        opencv = OpenCv()
        opencv.Q5_1()

    def show_hyperparameters(self):
        print('Show hyperparameters')
        opencv = OpenCv()
        opencv.Q5_2()
    
    def show_model_strucuture(self):
        print('Show model strucuture')
        opencv = OpenCv()
        opencv.Q5_3()
    
    def show_accuracy(self):
        print('Show accuracy')
        opencv = OpenCv()
        opencv.Q5_4()

    def test(self):
        print('Test')
        opencv = OpenCv()
        opencv.Q5_5()

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
        print('calculate...')
        # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        
        for f_idx in range(1, 16):
            # Read image and converty to gray image
            img = cv2.imread('./Q1_Image/'+str(f_idx)+'.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.gray_img[f_idx-1] = gray
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                # Draw and display the corner
                cv2.drawChessboardCorners(img, (11, 8), corners, ret)
                # save corner images
                self.corner_img[f_idx-1] = img
            

    def Q1_1(self):
        print('Q1_1')
        for f_idx in range(1, 16):
            cv2.namedWindow('1.1 Find Corners: '+str(f_idx)+'.bmp', cv2.WINDOW_NORMAL)
            cv2.imshow('1.1 Find Corners: '+str(f_idx)+'.bmp', self.corner_img[f_idx-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Q1_2(self):
        print('Q1_2')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[14].shape[::-1], None, None)
        print(mtx)

    def Q1_3(self, img_idx):
        print('Q1_3: '+str(img_idx+1)+'.bmp')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[img_idx].shape[::-1], None, None)
        R = cv2.Rodrigues(rvecs[img_idx])[0]
        t = tvecs[img_idx]
        extrinsic_matrix = np.hstack([R, t])
        print(extrinsic_matrix)

    def Q1_4(self):
        print('Q1_4')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray_img[0].shape[::-1], None, None)
        print(dist)

    def Q2(self):
        print('Q2')
        # Prepare object points, like (0, 0, 0), (1, 0, 0), ..., (7, 10, 0)
        objp = np.zeros((8*11, 3), np.float32)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
        objpoints = []                             # 3d point in real world space
        imgpoints = []                             # 2d points in image plane

        # Calculate camera calibration coefficients
        for f_idx in range(1, 6):
            # Read image and converty to gray image
            img = cv2.imread('./Q2_Image/'+str(f_idx)+'.bmp')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (11, 8), None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        cv2.namedWindow('Augmented Reality', cv2.WINDOW_NORMAL)
        # Draw a tetrahedron on chessboard
        for f_idx in range(1, 6):
            # Read image and converty to gray image
            img = cv2.imread('./Q2_Image/'+str(f_idx)+'.bmp')
            red = [0, 0, 255]   # BGR
            # tetrahefron cooridinates
            tetrahefron_coor = np.array([(3, 3, -3),
                                         (3, 5, 0),
                                         (5, 1, 0),
                                         (1, 1, 0)], dtype=np.float32)
            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(tetrahefron_coor, rvecs[f_idx-1], tvecs[f_idx-1], mtx, dist)
            imgpts = np.int32(imgpts).reshape(-1, 2)
            # draw the tetrahefron 
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[1]), red, 10)
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[2]), red, 10)
            cv2.line(img, tuple(imgpts[0]), tuple(imgpts[3]), red, 10)
            
            cv2.line(img, tuple(imgpts[1]), tuple(imgpts[2]), red, 10)
            cv2.line(img, tuple(imgpts[2]), tuple(imgpts[3]), red, 10)
            cv2.line(img, tuple(imgpts[3]), tuple(imgpts[1]), red, 10)

            cv2.imshow('Augmented Reality', img)
            cv2.waitKey(500)
        cv2.destroyAllWindows()

    def Q3(self):
        print('Q3')
        imgL = cv2.imread('./Q3_Image/'+'imL.png', 0)
        imgR = cv2.imread('./Q3_Image/'+'imR.png', 0)

        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        plt.imshow(disparity,'gray')
        plt.show()
    
    def Q4_1(self):
        print('Q4_1')
        img1 = cv2.imread('./Q4_Image/'+'Aerial1.jpg', 0)
        img2 = cv2.imread('./Q4_Image/'+'Aerial2.jpg', 0)
        
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # Sort keypoints in the order of their size, and select the first 6
        # Size is the region around a point of interest that is used to form the description of the keypoint
        kp1_knn = sorted(kp1, key=lambda x: x.size, reverse=True)[:7]
        kp2_knn = sorted(kp2, key=lambda x: x.size, reverse=True)[:7]
        print('kp1_knn')
        for i in range(0, 7):
            print('idx: {}, size: {}'.format([x.size for x in kp1].index(kp1_knn[i].size), kp1_knn[i].size))
        print('kp2_knn')
        for i in range(0, 7):
            print('idx: {}, size: {}'.format([x.size for x in kp2].index(kp2_knn[i].size), kp2_knn[i].size))
        # Draw keypoints
        img1_key = cv2.drawKeypoints(img1, kp1_knn, img1)
        img2_key = cv2.drawKeypoints(img2, kp2_knn, img2)

        cv2.namedWindow('Q3_1 Aerial1', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Q3_1 Aerial2', cv2.WINDOW_NORMAL)
        # Show image
        cv2.imshow('Q3_1 Aerial1', img1_key)
        cv2.imshow('Q3_1 Aerial2', img2_key)
        cv2.waitKey(0)
        # Write image
        cv2.imwrite('FeatureAerial1.jpg', img1_key)
        cv2.imwrite('FeatureAerial2.jpg', img2_key)
        cv2.destroyAllWindows()

    def Q4_2(self):
        print('Q4_2')
        img1 = cv2.imread('./Q4_Image/'+'Aerial1.jpg',0)
        img2 = cv2.imread('./Q4_Image/'+'Aerial2.jpg',0)

        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()
        # Find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        # Sort keypoints in the order of their size, and select the first 6
        # Size is the region around a point of interest that is used to form the description of the keypoint
        kp1_knn = sorted(kp1, key=lambda x: x.size, reverse=True)[:7]
        kp2_knn = sorted(kp2, key=lambda x: x.size, reverse=True)[:7]
        print('kp1_knn')
        for i in range(0, 7):
            print('idx: {}, size: {}'.format([x.size for x in kp1].index(kp1_knn[i].size), kp1_knn[i].size))
        print('kp2_knn')
        for i in range(0, 7):
            print('idx: {}, size: {}'.format([x.size for x in kp2].index(kp2_knn[i].size), kp2_knn[i].size))
        # Delete duplicated keypoints
        del kp1_knn[1]
        del kp2_knn[2]

        # Select the first 6 des according keypoints
        des1_knn = []
        des2_knn = []
        for i in range(len(kp1_knn)):
            idx1 = [x.size for x in kp1].index(kp1_knn[i].size)
            idx2 = [y.size for y in kp2].index(kp2_knn[i].size)
            des1_knn.append(des1[idx1])
            des2_knn.append(des2[idx2])
        des1_knn = np.asarray(des1_knn)
        des2_knn = np.asarray(des2_knn)
        
        # Create BFMatcher object
        bf = cv2.BFMatcher()
        # Match descriptors
        matches = bf.knnMatch(des1_knn, des2_knn, k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv2.drawMatchesKnn expects list of lists as matches.
        img3 = cv2.drawMatchesKnn(img1, kp1_knn, img2, kp2_knn, good, None, flags=2)
        cv2.namedWindow('Q3_2', cv2.WINDOW_NORMAL)
        cv2.imshow('Q3_2', img3)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def Q5_1(self):
        print('Q5_1')
    
    def Q5_2(self):
        print('Q5_2')
    
    def Q5_3(self):
        print('Q5_3')
    
    def Q5_4(self):
        print('Q5_4')

    def Q5_5(self):
        print('Q5_5')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())