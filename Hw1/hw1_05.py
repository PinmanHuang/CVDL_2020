"""
Author  : Joyce
Date    : 2020-10-28
"""
from UI.hw1_05_ui import Ui_MainWindow
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
import sys
import numpy as np
from matplotlib import pyplot as plt
import cv2
# Q5
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# Q5_1
import pickle
import random
# Q5_3
# from torchvision import models
from torchsummary import summary

# CUDA
CUDA_DEVICE_IDEX = 0
BATCH_SIZE = 32
LEARNING_RATE = 1e-2
EPOCHES = 80
OPTIMIZER = 'SGD'
USING_MODEL = 'VGG16_'+OPTIMIZER+'MOM_'+str(BATCH_SIZE)+'_'+str(LEARNING_RATE)

class PyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(PyMainWindow, self).__init__()
        self.setupUi(self)
        self.test_image_idx = ""
        self.pushButton_10.clicked.connect(self.show_train_images)
        self.pushButton_11.clicked.connect(self.show_hyperparameters)
        self.pushButton_12.clicked.connect(self.show_model_strucuture)
        self.pushButton_13.clicked.connect(self.show_accuracy)
        self.pushButton_14.clicked.connect(self.test)
        self.lineEdit.textChanged.connect(self.test_image_changed)

    def show_train_images(self):
        # print('Show train images')
        model = Model()
        model.Q5_1()

    def show_hyperparameters(self):
        # print('Show hyperparameters')
        model = Model()
        model.Q5_2()
    
    def show_model_strucuture(self):
        # print('Show model strucuture')
        model = Model()
        model.Q5_3()
    
    def show_accuracy(self):
        # print('Show accuracy')
        model = Model()
        # model.train()
        model.Q5_4()

    def test(self):
        # print('Test')
        model = Model()
        model.Q5_5(self.test_image_idx)

    def test_image_changed(self, text):
        self.test_image_idx = text
    
class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            #1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            #2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            #4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #6
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            #7
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #8
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #9
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #10
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            #13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.classifier = nn.Sequential(
            #14
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #15
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            #16
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        # out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

class Model(object):
    def __init__(self):
        super(Model, self).__init__()
    '''
    batches.meta: size is 10,
                  0 to 9 class,
                  each value is a label to an image
    data_batch_1 ~ data_batch_5: size is 10000*3072 (32*32*3 RGB),
                                 uint8 ndarray,
                                 each row is an image(32*32)
    '''
    def Q5_1(self):

        # Load data
        with open('./cifar-10-batches-py/batches.meta', 'rb') as fo:
            label_dict = pickle.load(fo, encoding='bytes')
        labels = label_dict[b'label_names']
        with open('./cifar-10-batches-py/data_batch_'+str(random.randint(1, 5)), 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
        X = data_dict[b'data']
        Y = data_dict[b'labels']
        # Generate random sample image index
        select_img_idx = random.sample(range(10000), 10)

        # Show images
        fig = plt.figure()
        ax = []
        for i in range(10):
            img_data = X[select_img_idx[i], :]                      # size is 3072
            img = img_data.reshape(3, 32, 32).transpose([1, 2, 0])  # reshape image to 3 tunnel(RGB)
            ax.append(fig.add_subplot(2, 5, i + 1))
            ax[-1].set_title(labels[Y[select_img_idx[i]]].decode())
            plt.imshow(img)
        plt.show()

    
    def Q5_2(self):
        print('hyperparameters:\nbatch size: '+str(BATCH_SIZE)+'\nlearning rate: '+str(LEARNING_RATE)+'\noptimizer: '+OPTIMIZER)
    
    def Q5_3(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device: {}'.format(device))
        model = VGG16().to(device)
        # model = models.vgg16()
        summary(model, (3, 32, 32))
    
    def Q5_4(self):
        img = cv2.imread('./Q5_Result/'+USING_MODEL+'.png')
        cv2.namedWindow(USING_MODEL, cv2.WINDOW_NORMAL)
        cv2.imshow(USING_MODEL, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def train(self):
        # Accuracy and loss data
        train_accuracy = []
        test_accuracy = []
        train_loss = []
        test_loss = []
        # Transform
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  # Normalize the test set same as training set without augmentation

        # Load data
        # train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=torchvision.transforms.ToTensor())
        train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=False, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        # test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=torchvision.transforms.ToTensor())
        test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2)
        # Create VGG16 model
        device = torch.device("cuda:"+str(CUDA_DEVICE_IDEX) if torch.cuda.is_available() else "cpu")
        print('Device: {}'.format(device))
        model = VGG16().to(device)
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)    # SGD
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)    # SGD-Momentum-Nesterov
        # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)   # Adam
        # Train the model
        for epoch in range(EPOCHES):
            running_loss = 0.0
            correct_pred = 0
            for i, data in enumerate(train_loader):
                img, label = data
                img = img.to(device)
                label = label.to(device)
                # Forwarding
                y_pred = model(img)
                _, pred = torch.max(y_pred, 1)          # prediction the index of the maximum value location
                correct_pred += (pred == label).sum()
                # Backpropagation
                loss = criterion(y_pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item())
            
            # Evaluate model
            model.eval()
            eval_loss = 0
            correct_eval = 0
            for j, eval_data in enumerate(test_loader):
                img2, label2 = eval_data
                img2 = img2.to(device)
                label2 = label2.to(device)
                # Predicting
                y_pred2 = model(img2)
                _, pred2 = torch.max(y_pred2, 1)          # prediction the index of the maximum value location
                correct_eval += (pred2 == label2).sum()
                # Loss
                loss2 = criterion(y_pred2, label2)
                eval_loss += float(loss2.item())

            train_accuracy.append(correct_pred.item() / (BATCH_SIZE * (i + 1)) * 100)
            test_accuracy.append(correct_eval.item() / (BATCH_SIZE * (j + 1)) * 100)
            train_loss.append(running_loss / (i + 1))
            test_loss.append(eval_loss / (j + 1))
            print('epoch {}/{}\tTrain loss: {:.4f}\tTrain accuracy: {:.2f}%\tTest loss: {:.4f}\tTest accuracy: {:.2f}%'.format(
                epoch + 1,
                EPOCHES,
                running_loss / (i + 1),
                correct_pred.item() / (BATCH_SIZE * (i + 1)) * 100,
                eval_loss / (j + 1),
                correct_eval.item() / (BATCH_SIZE * (j + 1)) * 100)
            )
            print()
        torch.save(model.state_dict(), USING_MODEL+'.pth')
        fig = plt.figure()
        plt.plot(train_accuracy)
        plt.plot(test_accuracy)
        plt.title('Accuracy')
        plt.xlabel('epoch')
        plt.ylabel('%')
        plt.legend(['Training', 'Testing'], loc='lower right')
        plt.show()

        fig = plt.figure()
        plt.plot(train_loss)
        plt.title('Loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.show()

    def Q5_5(self, test_idx):

        # Show image
        def imshow(img, result_list, class_list, label):
            img = img / 2 + 0.5     # unnormalize
            npimg = img.numpy()
            plt.subplot(2, 1, 1)
            plt.title('G.T.: '+class_list[label])
            plt.imshow(np.transpose(npimg, (1, 2, 0)))
            plt.subplot(2, 1, 2)
            plt.bar(range(10), result_list, align='center')
            plt.xticks(range(10), class_list, fontsize=10, rotation=45)
            plt.yticks((0.2, 0.4, 0.6, 0.8, 1))
            plt.show()

        if int(test_idx) <0 or int(test_idx) > 9999:
            print('Testing image index out of range.')
            return
        classes = ['plane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        # Test Dataset
        transform_test = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])  # Normalize the test set same as training set without augmentation
        test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
        # test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=torchvision.transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0)
        dataiter = iter(test_loader)
        for i in range(int(test_idx)):
            dataiter.next()
        image, label = dataiter.next()

        # Load Model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Device: {}'.format(device))
        model = VGG16().to(device)
        model.load_state_dict(torch.load('./model/'+USING_MODEL+'.pth', map_location=device))
        model.eval()
        # Evaluate
        output = model(image)
        softmax = nn.Softmax()
        probability = softmax(output).tolist()[0]
        # show images
        imshow(torchvision.utils.make_grid(image), probability, classes, label)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = PyMainWindow()
    ui.setupUi(window)
    ui.show()
    sys.exit(app.exec_())