"""
Author  : Joyce
Date    : 2020-12-25
"""
from UI.hw2_05_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
from glob import glob
import random
import os, datetime
# Q5
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

class PyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        self.pushButton.clicked.connect(self.train)
        self.pushButton_2.clicked.connect(self.tensorboard)
        self.pushButton_3.clicked.connect(self.random_select)
        self.pushButton_4.clicked.connect(self.improve)

    def train(self):
        model = Model()
        model.Q5_1()

    def tensorboard(self):
        model = Model()
        model.Q5_2()

    def random_select(self):
        model = Model()
        model.Q5_3()

    def improve(self):
        model = Model()
        model.Q5_4()

class Resnet50(object):
    def __init__(self):
        super(Resnet50, self).__init__()

    def model(self):
        input_shape = IMAGE_SIZE + (3,)
        inputs = keras.Input(shape=input_shape)
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        res_blocks = [3, 4, 6, 3]
        res_filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]

        first_conv = 1
        for index, block in enumerate(res_blocks):  # 0, 3
            for layer in range(block):  # 3
                input_tensor = x
                for idx, f in enumerate(res_filters[index]):
                    pad = 'valid'
                    ksize = (1, 1)
                    if idx > 0 and idx < 2:
                        ksize = (3, 3)
                        pad = 'same'

                    strides = (1, 1)
                    if first_conv == 1:
                        first_conv = 0

                    elif idx == 0 and layer == 0:
                        strides = (2, 2)

                    x = layers.Conv2D(f, ksize, strides=strides, kernel_initializer='he_normal', padding=pad)(x)
                    #             print(block, layer, f, ksize, strides, pad)
                    x = layers.BatchNormalization()(x)
                    if idx < 2:
                        x = layers.Activation("relu")(x)

                if layer == 0:
                    strides = (2, 2)
                    if index == 0:
                        strides = (1, 1)

                    shortcut = layers.Conv2D(res_filters[index][-1], (1, 1), strides=strides,
                                            kernel_initializer='he_normal')(input_tensor)
                    shortcut = layers.BatchNormalization()(shortcut)
                else:
                    #             print('i', ksize, strides)
                    shortcut = input_tensor

                x = layers.add([x, shortcut])
                x = layers.Activation('relu')(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1, activation='sigmoid')(x)
        outputs = x
        resnet50 = keras.Model(inputs, outputs)
        return resnet50

class Model(object):
    def __init__(self):
        super(Model, self).__init__()
    
    def data_generator(self):
        train_datagen = ImageDataGenerator(rescale=1/255)
        val_datagen = ImageDataGenerator(rescale=1/255)
        train_generator = train_datagen.flow_from_directory(
            'Q5_Dataset/train', 
            class_mode = 'binary', 
            target_size = IMAGE_SIZE,
            batch_size = BATCH_SIZE,
            shuffle = True
        )
        validation_generator  = val_datagen.flow_from_directory(
            'Q5_Dataset/validation', 
            class_mode = 'binary', 
            target_size = IMAGE_SIZE, 
            batch_size = BATCH_SIZE,
            shuffle = True
        )
        test_generator  = val_datagen.flow_from_directory(
            'Q5_Dataset/test', 
            class_mode = 'binary', 
            target_size = IMAGE_SIZE, 
            batch_size = 1,
            shuffle = True
        )
        return train_generator, validation_generator, test_generator

    def Q5_1(self):
        print('Q5_1')
        train_generator, validation_generator, test_generator = self.data_generator()
        resnet50 = Resnet50()
        model = resnet50.model()
        model.compile(
            optimizer = keras.optimizers.Adam(1e-3),
            loss = "binary_crossentropy",
            metrics = ["accuracy"],
        )
        model.summary()

        epochs = 5

        logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

        callbacks = [
            tensorboard_callback,
        #     keras.callbacks.ModelCheckpoint("no_augmemt_save_at_{epoch}.h5"),
        ]

        history = model.fit(
            train_generator, epochs=epochs, callbacks=callbacks, validation_data=validation_generator,
            steps_per_epoch=500,validation_steps=50
        )
        model.save("resnet_CatDog_resizing.h5")
        

    def Q5_2(self):
        print('Q5_2')

    def Q5_3(self):
        print('Q5_3')
        _, _, test_generator = self.data_generator()
        plt.figure(figsize=(6, 6))
        labels = ['cat', 'dog']

        data, label = test_generator.next()
        img = (data[0] * 255).astype('uint8')

        plt.title('Class: '+labels[int(label[0])])
        plt.imshow(img)
        plt.show()

    def Q5_4(self):
        print('Q5_4')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())