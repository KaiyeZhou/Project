# -*- coding: UTF-8 -*-

import numpy as np
import scipy.io as sio
import time

def classify_count(data1):
    ### Use the coordinates of keypoints to classify and count actions.

    ### load coordinates
    ## N x 4 x 17 (frames, (x, y, logit, prob), 17keypoints)
    # data1 = sio.loadmat("/home/server010/server010/FitNess/FitNess_datas/2019_04_11_test/coordinate_mat/VID_20181204_161326.mat")['name'][:, 0:2, :]       #699x2x17
    # data1 = sio.loadmat("/home/server010/server010/FitNess/FitNess_datas/standard_action/out_video/wotui_zuohou.mat")['name'][:, 0:2, :]       #699x2x17

    # parameters
    temp1, temp2, temp3 = 0, 0, 0
    count1, count2, count3 = 0, 0, 0
    flag = 0
    start_frame = []
    end_frame = []
    maxx = 0
    maxList = []

    time_start = time.time()
    for i in range(len(data1)):   # i-(0, N), N frames
        ### dead lift
        if data1[i][1][9] > data1[i][1][7] and data1[i][1][10] > data1[i][1][8]:
            temp1 += 1
        
        ### deep squat
        if data1[i][1][9] < data1[i][1][7] and data1[i][1][10] < data1[i][1][8] and (abs(data1[i][1][11] - data1[i][1][13]) > 90 or abs(data1[i][1][12] - data1[i][1][14]) > 90):
            temp2 += 1
        
        ### bench press
        if abs(data1[i][1][5] - data1[i][1][13]) < 100 or abs(data1[i][1][6] - data1[i][1][14]) < 100:
            temp3 += 1
    
    ### dead lift  
    if temp1 > len(data1)/2:
        for i in range(len(data1)):
            if flag == 0:
                if i + 4 < len(data1) and data1[i][1][9] - data1[i + 4][1][9] > 90:
                    flag = 1
                    # if start_frame and i - start_frame[-1] > 30:
                    start_frame.append(i)
                
            if flag == 1:
                if i + 4 < len(data1) and abs(data1[i + 4][1][9] - data1[start_frame[0]][1][9]) < 20:                    
                    flag = 0
                    count1 += 1
                    maxList.append(maxx)
                    maxx = 0
                if data1[start_frame[0]][1][9] - data1[i][1][9] > maxx:
                    maxx = data1[start_frame[0]][1][9] - data1[i][1][9]
       
        a = 0
        # print('He/She is doing dead lift')
        # print('counts: %d' %count1)
        return a, count1, maxList
        # print(start_frame)
        # print(maxList)
        # for i in range(len(maxList)):
        #         if maxList[i] > 260:
        #             print('第%d 个动作完成度： 100 %%' %(i + 1))
        #         else:
        #             percent = maxList[i] / 260 * 100
        #             print('第%d 个动作完成度： %d %%' %(i + 1, int(percent)))
        # if min(maxList) > 260:
        #     print('整体评分： 100 分')
        # else:
        #     print('整体评分: %d 分' %(int(min(maxList) / 260 * 100)))
    
    ### deep squat
    if temp2 > len(data1)/2:
        for i in range(len(data1)):
            if flag == 0:
                if i + 4 < len(data1) and data1[i + 4][1][9] - data1[i][1][9] > 90:
                    flag = 1        
                    start_frame.append(i)
            if flag == 1:
                if i + 4 < len(data1) and abs(data1[i + 4][1][9] - data1[start_frame[-1]][1][9]) < 20:            
                    flag = 0
                    count2 += 1
                    maxList.append(maxx)
                    maxx = 0
                if data1[i][1][9] - data1[start_frame[-1]][1][9] > maxx:
                    maxx = data1[i][1][9] - data1[start_frame[-1]][1][9]
        
        b = 1
        # print('He/She is doing deep squat')
        # print('counts: %d' %count2)
        return b, count2, maxList
        # print(start_frame)
        # print(maxList)
        # for i in range(len(maxList)):
        #         if maxList[i] > 160:
        #             print('第%d 个动作完成度： 100 %%' %(i + 1))
        #         else:
        #             percent = maxList[i] / 160 * 100
        #             print('第%d 个动作完成度： %d %%' %(i + 1, int(percent)))
        # if min(maxList) > 160:
        #     print('整体评分： 100 分')
        # else:
        #     print('整体评分: %d 分' %(int(min(maxList) / 160 * 100)))
    
    ### bench press
    if temp3 > 10:
        for i in range(len(data1)):
            if flag == 0:
                if i + 4 < len(data1) and (data1[i + 4][1][9] - data1[i][1][9] > 120 or data1[i + 4][1][10] - data1[i][1][10] > 120):
                    flag = 1        
                    start_frame.append(i)
            if flag == 1:
                if i + 4 < len(data1) and (abs(data1[i + 4][1][9] - data1[start_frame[-1]][1][9]) < 5 or abs(data1[i + 4][1][10] - data1[start_frame[-1]][1][10]) < 5):
                    flag = 0
                    count3 += 1
                    maxList.append(maxx)
                    maxx = 0
                if data1[i][1][9] - data1[start_frame[-1]][1][9] > maxx:
                    maxx = data1[i][1][9] - data1[start_frame[-1]][1][9]
                # if data1[i][1][10] - data1[start_frame[-1]][1][10] > maxx:
                #     maxx = data1[i][1][10] - data1[start_frame[-1]][1][10]
        
        c = 2
        # print('He/She is doing bench press')
        # print('counts: %d' %count3)
        return c, count3, maxList
        # print(start_frame)
        # print(maxList)
        # for i in range(len(maxList)):
        #         if maxList[i] > 130:
        #             print('第%d 个动作完成度： 100 %%' %(i + 1))
        #         else:
        #             percent = maxList[i] / 130 * 100
        #             print('第%d 个动作完成度： %d %%' %(i + 1, int(percent)))
        # # if min(maxList) > 130:
        # #     print('整体评分： 100 分')
        # # else:
        # #     print('整体评分: %d 分' %(int(min(maxList) / 130 * 100)))
        # print('整体评分: %d 分' %(int(sum(maxList) / (130 * len(maxList)) * 100)))
        
    
    
    time_end = time.time()
    # print('totally cost: {:.3f}s'.format(time_end - time_start))



if __name__ == "__main__":
    classify_count()