# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from PyQt5.QtWidgets import *
import os
from Ui_Fitness import Ui_MainWindow

from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog,QTabWidget 
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel,QWidget
import cv2, sys, time


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """
    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
    
    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        os.system('cheese')

    
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        self.label_2.setText('12345')
    
    def videoprocessing(self):
        print("gogo")
        global videoName #在这里设置全局变量以便在线程中使用
        videoName,videoType= QFileDialog.getOpenFileName(self,
                                    "打开视频",
                                    "",
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.mp4;;*.avi;;All Files (*)")
        #cap = cv2.VideoCapture(str(videoName))
        th = Thread(self)
        th.changePixmap.connect(self.setImage)
        th.start()
    
    def setImage(self, image):
        self.label_3.setPixmap(QPixmap.fromImage(image))
    
    def videoprocessing2(self):
        print("gogo")
        global videoName2 #在这里设置全局变量以便在线程中使用
        videoName2,videoType= QFileDialog.getOpenFileName(self,
                                    "打开视频",
                                    "",
                                    #" *.jpg;;*.png;;*.jpeg;;*.bmp")
                                    " *.mp4;;*.avi;;All Files (*)")
        #cap = cv2.VideoCapture(str(videoName))
        th2 = Thread2(self)
        th2.changePixmap.connect(self.setImage2)
        th2.start()

    def setImage2(self, image):
        self.label_4.setPixmap(QPixmap.fromImage(image))

class Thread(QThread):#采用线程来播放视频

    changePixmap = pyqtSignal(QtGui.QImage)
    def run(self):
        cap = cv2.VideoCapture(videoName)
        print(videoName)
        while (cap.isOpened()==True):
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(0.01) #控制视频播放的速度
            else:
                break

class Thread2(QThread):#采用线程来播放视频

    changePixmap = pyqtSignal(QtGui.QImage)
    def run(self):
        cap = cv2.VideoCapture(videoName2)
        print(videoName2)
        while (cap.isOpened()==True):
            ret, frame = cap.read()
            if ret:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)#在这里可以对每帧图像进行处理，
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(0.01) #控制视频播放的速度
            else:
                break



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    ui = MainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    ui.show()
    sys.exit(app.exec_())