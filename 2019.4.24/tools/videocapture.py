# -*- coding: utf-8 -*-

import cv2
import time  # 导入时间模块
import shutil

start_time = time.time() 
# t=time.ctime(c) #获得当前系统时间

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
