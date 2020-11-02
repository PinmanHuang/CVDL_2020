# Table of Contents
* [About the Homework 1](#about-the-homework-1)
    * [VGG16](#VGG16)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Installation](#installation)
# About the Homework 1
![Hw1](https://i.imgur.com/JGxIRC9.png)
![Q5](https://i.imgur.com/oCZRHUI.png)
## VGG16
Training cifar10 classifier using VGG16.
* Dataset: Cifar10
* [Model](https://drive.google.com/drive/folders/17_0pd40iayfL_n9jmDu-eSZJbS3GZESh?usp=sharing)
# Getting Started
## Prerequisites
* Python=3.7
* opencv-contrib-python=3.4.2.1
```sh
pip install opencv-contrib-python==3.4.2.17
```
* Matplotlib=3.1.1
```sh
pip install matplotlib==3.1.1
```
* pyqt5=5.15.1
```sh
pip install pyqt5
pip install PyQt5-sip
pip install pyqt5-tools
```
* pyqt5 using steps
    * find designer.exe and run
    * select Main Window to create new UI
    * using IDE to design your UI
    * save
    * transfer ui file into python file
        ```sh
        pyuic5 {PyQt_UI_FILE_NAME}.ui -o {PyQt_PY_FILE_NAME}.py
        ```
    * create main.py to show the ui
    * create executable file
        ```sh
        pyinstaller -F ./main.py
        ```
* [PyTorch](https://pytorch.org/get-started/locally/)
```sh
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
```
## Installation
1. Clone the repo
```sh
git clone https://github.com/PinmanHuang/CVDL_2020.git
```
2. Run the python project
```sh
python hw1.py
```