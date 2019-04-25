# -*- coding: UTF-8 -*-

import numpy as np
import scipy.io as sio
import time

def classify_count():
    ### Use the coordinates of keypoints to classify and count actions.

    ### load coordinates
    ## N x 4 x 17 (frames, (x, y, logit, prob), 17keypoints)
    data1 = sio.loadmat("/home/server010/server010/FitNess/FitNess_datas/2019_04_11_test/coordinate_mat/VID_20181204_161128.mat")['name'][:, 0:2, :]       #699x2x17
     
    # parameters
    temp1, temp2 = 0, 0
    count1, count2 = 0, 0
    flag = 0
    start_frame = []

    time_start = time.time()
    for i in range(len(data1)):   # i-(0, N), N frames
        ### dead lift
        if data1[i][1][9] > data1[i][1][7] and data1[i][1][10] > data1[i][1][8]:
            temp1 += 1
        
        ### deep squat
        if data1[i][1][9] < data1[i][1][7] and data1[i][1][10] < data1[i][1][8]:
            temp2 += 1
    
    ### dead lift  
    if temp1 > len(data1)/2:
        for i in range(len(data1)):
            if flag == 0:
                if i + 20 < len(data1) and data1[i][1][9] - data1[i+20][1][9] > 100:
                    flag = 1
                    # if start_frame and i - start_frame[-1] > 30:
                    start_frame.append(i)
            if flag == 1:
                if i + 20 < len(data1) and abs(data1[i+20][1][9] - data1[start_frame[0]][1][9]) < 15:                    
                    flag = 0
                    count1 += 1
        # a = 0
        print('He/She is doing dead lift')
        print('counts: %d' %count1)
        # return a, count1
        # print(start_frame)
    
    ### deep squat
    if temp2 > len(data1)/2:
        for i in range(len(data1)):
            if flag == 0:
                if i + 20 < len(data1) and data1[i+20][1][9] - data1[i][1][9] > 90:
                    flag = 1        
                    start_frame.append(i)
            if flag == 1:
                if i + 20 < len(data1) and abs(data1[i+20][1][9] - data1[start_frame[-1]][1][9]) < 20:            
                    flag = 0
                    count2 += 1                    
        # b = 1
        print('He/She is doing deep squat')
        print('counts: %d' %count2)
        # return b, count2
        # print(start_frame)
    
    time_end = time.time()
    # print('totally cost: {:.3f}s'.format(time_end - time_start))



if __name__ == "__main__":
    classify_count()