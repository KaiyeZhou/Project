# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import *
import os
from Ui_Fitness import Ui_MainWindow
from densepose import densepose


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
        action_classify, action_count = densepose()
        if action_classify == 0:
            self.textBrowser.append('正在做硬拉动作')
            self.textBrowser.append('做了%d 次' %action_count)
        
        if action_classify == 1:
            self.textBrowser.append('正在做深蹲动作')
            self.textBrowser.append('做了%d 次' %action_count)



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    # MainWindow = QtWidgets.QMainWindow()
    ui = MainWindow()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    ui.show()
    sys.exit(app.exec_())