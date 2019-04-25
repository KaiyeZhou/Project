# -*- coding: utf-8 -*-

import os
import cv2
import sys
import time
import shutil

from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import QFileDialog,QTabWidget 
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QLabel,QWidget

from Ui_Fitness import Ui_MainWindow
from densepose import densepose
from classify_count import classify_count
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default='/home/server010/server010/FitNess/densepose/configs/configs_use/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.yaml',
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default='/home/server010/server010/FitNess/densepose/configs/configs_use/DensePoseKeyPointsMask_ResNet50_FPN_s1x-e2e.pkl',
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/home/server010/server010/FitNess/Video_capture/video_out/',
        type=str
    )
    # parser.add_argument(
    #     '--image-ext',
    #     dest='image_ext',
    #     help='image file name extension (default: jpg)',
    #     default='jpg',
    #     type=str
    # )
    # parser.add_argument(
    #     'im_or_folder', help='image or folder of images', default=None
    # )
    parser.add_argument(
        '--video', help='input video', default='/home/server010/server010/FitNess/Video_capture/test1.mp4'
    )
    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)
        # print(parser.parse_args())
    return parser.parse_args()

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

    @pyqtSlot()
    def on_pushButton_2_clicked(self):
        # os.system('cheese')
        start_time = time.time() 

        savepath = '/home/server010/server010/FitNess/Video_capture/test1.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v'.encode('utf-8'))
        out = cv2.VideoWriter(savepath, fourcc, 30, (640,480))
        cap = cv2.VideoCapture(0)
        delays = 20
        # cv2.startWindowThread()
        while(1):
            ret,frame = cap.read()   #get frame
            cv2.imshow("Fitness", frame)
            out.write(frame)
            if cv2.waitKey(1) & 0xFF==ord('q') or ret==False or time.time() - start_time > delays:
                break
        cap.release()
        out.release()
        # cv2.destroyAllwindows()
        video_path = '/home/server010/server010/FitNess/save_video/' + time.strftime('%Y-%m-%d-%H-%M',time.localtime(time.time())) + '.mp4'
        shutil.copyfile(savepath, video_path)

    
    @pyqtSlot()
    def on_pushButton_3_clicked(self):
        time_start = time.time()
        self.textBrowser.setPlainText('')
        action_classify, action_count, maxList = densepose(parse_args())
        # action_classify, action_count = classify_count()
        # action_classify = 1
        # action_count = 2
        if action_classify == 0:
            self.textBrowser.append('正在做硬拉动作')
            self.textBrowser.append('做了%d 次' %action_count)
            for i in range(len(maxList)):
                if maxList[i] > 260:
                    self.textBrowser.append('第%d 个动作完成度： 100 %%' %(i + 1))
                else:
                    percent = maxList[i] / 260 * 100
                    self.textBrowser.append('第%d 个动作完成度： %d %%' %(i + 1, int(percent)))
            if min(maxList) > 260:
                self.textBrowser.append('整体评分： 100 分')
            else:
                self.textBrowser.append('整体评分: %d 分' %(int(min(maxList) / 260 * 100)))
        
        if action_classify == 1:
            self.textBrowser.append('正在做深蹲动作')
            self.textBrowser.append('做了%d 次' %action_count)
            for i in range(len(maxList)):
                if maxList[i] > 160:
                    self.textBrowser.append('第%d 个动作完成度： 100 %%' %(i + 1))
                else:
                    percent = maxList[i] / 160 * 100
                    self.textBrowser.append('第%d 个动作完成度： %d %%' %(i + 1, int(percent)))
            if min(maxList) > 160:
                self.textBrowser.append('整体评分： 100 分')
            else:
                self.textBrowser.append('整体评分: %d 分' %(int(min(maxList) / 160 * 100)))
        
        if action_classify == 2:
            self.textBrowser.append('正在做卧推动作')
            self.textBrowser.append('做了%d 次' %action_count)
            for i in range(len(maxList)):
                if maxList[i] > 130:
                    self.textBrowser.append('第%d 个动作完成度： 100 %%' %(i + 1))
                else:
                    percent = maxList[i] / 130 * 100
                    self.textBrowser.append('第%d 个动作完成度： %d %%' %(i + 1, int(percent)))
            self.textBrowser.append('整体评分: %d 分' %(int(sum(maxList) / (130 * len(maxList)) * 100)))
        time_end = time.time()
        self.textBrowser.append('totally cost: {:.3f}s'.format(time_end - time_start))

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
                p = convertToQtFormat.scaled(640, 360, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(0.03) #控制视频播放的速度
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
                p = convertToQtFormat.scaled(640, 360, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                time.sleep(0.03) #控制视频播放的速度
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