# -*- coding: utf-8 -*-

import cv2
import numpy as np

cap = cv2.VideoCapture('/home/server010/server010/FitNess/FitNess_datas/datasets/1204dataset/VID_20181204_161008.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('Replay', frame)
    if cv2.waitKey(40) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow()