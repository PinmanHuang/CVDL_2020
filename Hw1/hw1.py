"""
Author  : Joyce
Date    : 2020-10-12
"""
from UI.hw1_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys

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

    def find_intrinsic(self):
        print('find intrinsic')

    def find_distortion(self):
        print('find distortion')
    
    def find_extrinsic(self):
        print('find extrinsic')

    def select_image(self, text):
        print('select image: {}'.format(text))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())