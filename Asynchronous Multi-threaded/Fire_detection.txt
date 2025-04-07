#!/usr/bin/env python
# -*- coding:utf-8 -*-
import cv2 as cv
import numpy as np
w_thre=50; #长宽最小阈值
h_thre=50;
def find_countours_center(frame,countours):
    C=[]#放中心点
    for i in range(np.array(contours).shape[0]):
        # print(countours[i])
        x_list = []
        y_list = []
        for j in range(np.array(countours[i]).shape[0]):#将一个闭合曲线里面的所有x,y轴坐标放在两个list里
            x_list.append(countours[i][j][0][0])
            y_list.append(countours[i][j][0][1])
        if (max(x_list)-min(x_list))>w_thre or (max(y_list)-min(y_list))>h_thre:
            x=min(x_list);y=min(y_list);h=max(y_list)-min(y_list);w=-min(x_list)+max(x_list);
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 100, 255), 2);
            C.append([(min(x_list)+max(x_list))/2,(max(y_list)+min(y_list))/2]); #找到闭合曲线中点位置
def contrast_brightness_demo(image, c, b): #其中c为对比度，b为每个像素加上的值（调节亮度）
    blank = np.zeros(image.shape, image.dtype) #创建一张与原图像大小及通道数都相同的黑色图像
    dst = cv.addWeighted(image, c, blank, 1-c, b) #c为加权值，b为每个像素所加的像素值
    ret, dst = cv.threshold(dst, 25, 255, cv.THRESH_BINARY)
    return dst
capture=cv.VideoCapture("E:\\FFOutput\\fire01.avi")
redThre = 150
saturationTh = 50 #饱和度设置
while(True):
    ret, frame = capture.read()
    B = frame[:, :, 0]
    G = frame[:, :, 1]
    R = frame[:, :, 2]
    minValue = np.array(np.where(R <= G, np.where(G <= B, R, np.where(R <= B, R, B)), np.where(G <= B, G, B)))
    S = 1 - 3.0 * minValue / (R + G + B + 1)
    fireImg = np.array(np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S >= 0.2, np.where(S >= (255 - R) * saturationTh / redThre, 255, 0), 0), 0), 0), 0))
    gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
    gray_fireImg[:, :, 0] = fireImg
    gray_fireImg = cv.GaussianBlur(gray_fireImg, (7, 7), 0) #函数用高斯滤波器（GaussianFilter）对图像进行平滑处理。
    gray_fireImg = contrast_brightness_demo(gray_fireImg, 5.0, 25)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))#膨胀就是图像中的高亮部分进行膨胀，“领域扩张”，效果图拥有比原图更大的高亮区域；腐蚀就是原图中的高亮部分被腐蚀，“领域被蚕食”，效果图拥有比原图更小的高亮区域
    gray_fireImg = cv.morphologyEx(gray_fireImg, cv.MORPH_CLOSE, kernel) #闭操作，先膨胀后腐蚀
    white_array=np.where(gray_fireImg==1);
    dst = cv.bitwise_or(frame, frame, mask=gray_fireImg);

    ###############画框圈火焰##############
    _,contours, hierarchy = cv.findContours(gray_fireImg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE); #CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE
    len_contours=np.array(contours).shape;
    find_countours_center(frame,contours)
    cv.drawContours(frame, contours, -1, (0, 255, 0), 3);
    cv.imshow("frame", frame)
    cv.imshow("fire", dst)
    cv.imshow("gray_fireImg", gray_fireImg)
    c = cv.waitKey(40)
    if c == 27: #ESC退出
        break

