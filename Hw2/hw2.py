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
# Q3
import cv2.aruco as aruco
from matplotlib import pyplot as plt
# Q4
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

class PyMainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        # === push button clicked action === 
        self.pushButton.clicked.connect(self.bg_sub)
        self.pushButton_2.clicked.connect(self.preprocessing)
        self.pushButton_3.clicked.connect(self.tracking)
        self.pushButton_4.clicked.connect(self.transform)
        self.pushButton_5.clicked.connect(self.reconstruction)
        self.pushButton_6.clicked.connect(self.error)

    def bg_sub(self):
        opencv = OpenCv()
        opencv.Q1_1()

    def preprocessing(self):
        opencv = OpenCv()
        opencv.Q2_1()

    def tracking(self):
        opencv = OpenCv()
        opencv.Q2_2()

    def transform(self):
        opencv = OpenCv()
        opencv.Q3_1()

    def reconstruction(self):
        opencv = OpenCv()
        opencv.Q4_1()
    
    def error(self):
        opencv = OpenCv()
        opencv.Q4_2()

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

    def Q3_1(self):
        cv2.namedWindow('3.1 Perspective Transform', cv2.WINDOW_NORMAL)
        # read image
        im_src = cv2.imread("./Q3_Image/rl.jpg")
        size = im_src.shape
        pts_src = np.array([
            [0, 0],
            [size[1], 0],
            [size[1], size[0]],
            [0, size[0]]
        ], dtype=float)

        # capture video and get fps
        cap = cv2.VideoCapture('./Q3_Image/test4perspective.mp4')
        fps = cap.get(cv2.CAP_PROP_FPS)
        # get 1st frame of video
        ret, frame = cap.read()
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./Q3_Result/3_1_result.mp4', fourcc, 20.0, (frame.shape[1], frame.shape[0]))

        while(cap.isOpened()):
            # get frames of video and convert to gray
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # reading the four code and get its idx and corner
            aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
            arucoParameters = aruco.DetectorParameters_create()
            corners, ids, rejectedImgPoints = aruco.detectMarkers(
                gray,
                aruco_dict,
                parameters = arucoParameters
            )

            # doesn't have four corners
            if len(corners) != 4:
                continue

            # have four corners
            if np.all(ids != None):
                # index of each markers
                idx_23 = np.where(ids==23)[0][0]
                idx_25 = np.where(ids==25)[0][0]
                idx_30 = np.where(ids==30)[0][0]
                idx_33 = np.where(ids==33)[0][0]
                # get four point location
                p1 = (corners[idx_25][0][1][0], corners[idx_25][0][1][1])
                p2 = (corners[idx_33][0][2][0], corners[idx_33][0][2][1])
                p3 = (corners[idx_30][0][0][0], corners[idx_30][0][0][1])
                p4 = (corners[idx_23][0][0][0], corners[idx_23][0][0][1])
                # calculate distance and scale the point location
                distance = np.linalg.norm(np.subtract(p1, p2))
                scaling = round(0.02 * distance)
                pts_dst = np.array([
                    [p1[0] - scaling, p1[1] - scaling],
                    [p2[0] + scaling, p2[1] - scaling],
                    [p3[0] + scaling, p3[1] + scaling],
                    [p4[0] - scaling, p4[1] + scaling]
                ])

                # frame image
                im_dst = frame

                # find homography and warp perspective
                h, status = cv2.findHomography(pts_src, pts_dst)
                temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
                
                # draw
                cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16)
                im_dst = im_dst + temp
                out.write(im_dst)
                cv2.imshow('3.1 Perspective Transform', im_dst)
            
            if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
                break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    def norm(self, img):
        img = img.copy()
        img -= np.min(img)
        img /= np.max(img)
        img *= 255.
        return np.uint8(img)

    def reconstruction(self, img):
        # split 3 channels
        b, g, r = cv2.split(img)
        # PCA
        pca = PCA(25)
        # b
        # b_norm = normalize(b)
        lower_dimension_b = pca.fit_transform(b)
        approximation_b = pca.inverse_transform(lower_dimension_b)
        # g
        # g_norm = normalize(g)
        lower_dimension_g = pca.fit_transform(g)
        approximation_g = pca.inverse_transform(lower_dimension_g)
        # r
        # r_norm = normalize(r)
        lower_dimension_r = pca.fit_transform(r)
        approximation_r = pca.inverse_transform(lower_dimension_r)
        
        # clip
        clip_b = np.clip(approximation_b, a_min = 0, a_max = 255)
        clip_g = np.clip(approximation_g, a_min = 0, a_max = 255)
        clip_r = np.clip(approximation_r, a_min = 0, a_max = 255)
        # merge
        n_img = (cv2.merge([clip_b, clip_g, clip_r])).astype(np.uint8)
        # n_img = (np.dstack((approximation_b, approximation_g, approximation_r))).astype(np.uint8)
        # n_img = (cv2.merge([approximation_b, approximation_g, approximation_r])).astype(int)
        # n_img = cv2.normalize(n_img, None, 0, 255, cv2.NORM_MINMAX)
        # np_img = n_img.astype(int)
        # n_img = norm(np.float32(n_img))
        '''
        X = img.reshape((100, -1))
        pca = PCA(25)
        lower_dimension = pca.fit_transform(X)
        approximation = pca.inverse_transform(lower_dimension)

        approximation = approximation.reshape(100, 100, 3)
        n_img = norm_image(approximation)
        '''
        return n_img

    def Q4_1(self):
        fig = plt.figure(figsize=(18, 4))
        for idx in range(1, 18, 1):
            # ====== i ======
            img = cv2.imread('./Q4_Image/'+str(idx)+'.jpg')[:, :, ::-1]
            n_img = self.reconstruction(img)
            # plot
            # origin image
            plt.subplot(4, 17, idx)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            # reconstruction image
            plt.subplot(4, 17, idx + 17)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(n_img)
            # ===============

            # ====== i+17 ======
            img = cv2.imread('./Q4_Image/'+str(idx + 17)+'.jpg')[:, :, ::-1]
            n_img = self.reconstruction(img)
            # plot
            # origin image
            plt.subplot(4, 17, idx + 34)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)
            # reconstruction image
            plt.subplot(4, 17, idx + 51)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(n_img)
            # ==================
        fig.text(0, 0.9, 'Original', va='center', rotation='vertical')
        fig.text(0, 0.65, 'Reconstruction', va='center', rotation='vertical')
        fig.text(0, 0.4, 'Original', va='center', rotation='vertical')
        fig.text(0, 0.15, 'Reconstruction', va='center', rotation='vertical')
        plt.tight_layout(pad=0.5)

        plt.show()

    def Q4_2(self):
        for idx in range(1, 35, 1):
            img = cv2.imread('./Q4_Image/'+str(idx)+'.jpg')
            n_img = self.reconstruction(img)
            # convert to gray
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            n_img_gray = cv2.cvtColor(n_img, cv2.COLOR_BGR2GRAY)
            n_img_gray = cv2.normalize(n_img_gray, None, 0, 255, cv2.NORM_MINMAX)
            error = np.sum(np.absolute(img_gray-n_img_gray))
            print('{}: {}'.format(idx, error))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())