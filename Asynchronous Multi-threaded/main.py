import cv2
import time
import numpy as np
import serial
import argparse
import sys
import os
from rknnpool import rknnPoolExecutor
# 图像处理函数，实际应用过程中需要自行修改
from func import myFunc, global_result, global_serial

# cap = cv2.VideoCapture('./video/firevideo.mp4')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

modelPath = "/home/orangepi/Desktop/704/serial/1919example/rknnModel/best1920.rknn"
# 线程数
TPEs = 6
result_string_later = ""
# 初始化rknn池
pool = rknnPoolExecutor(
    rknnModel=modelPath,
    TPEs=TPEs,
    func=myFunc)

# 初始化异步所需要的帧
if (cap.isOpened()):
    for i in range(TPEs + 1):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            del pool
            exit(-1)
        pool.put(frame)

frames, loopTime, initTime = 0, time.time(), time.time()

parser = argparse.ArgumentParser(description='')
parser.add_argument("--device", type=str, default="/dev/ttyS1", help='specify the serial node')
args = parser.parse_args()

try:
    ser = serial.Serial(args.device, 115200, timeout=1)
    ser.flush()
except serial.SerialException:
    print("Unable to open serial device: %s" % args.device)
    sys.exit(-1)

ser.write(b'POSReady\r\n')  # 发送启动指令

while (cap.isOpened()):
    frames += 1
    ret, frame = cap.read()
    if not ret:
        break

    pool.put(frame)
    frame, flag = pool.get()

    if flag == False:
        break

    if ser.in_waiting > 0:  # 检查串口是否有数据
        data = ser.read(1).decode('utf-8')  # 读取一个字节并解码

        # 打印接收到的数据
        print(f"接收到数据: {data}")

        if data == 'w':  # 检查是否接收到 'w'
            ser.write(b'OK\r\n')  # 发送 'OK' 到串口
            sys.exit()

    result_string_before = ''.join(str(elem) for elem in global_result)
    print("result_string_before：", result_string_before)
    
        
    if result_string_before != result_string_later:
        ser.write(('POS' + result_string_before + '\r\n').encode())

    result_string_later = result_string_before

    frame = cv2.resize(frame, (1280, 640))
    cv2.imshow('Combined Image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if frames % 30 == 0:
        print("30帧平均帧率:	", 30 / (time.time() - loopTime), "帧")
        loopTime = time.time()

print("总平均帧率	", frames / (time.time() - initTime))
# 释放cap和rknn线程池
cap.release()
cv2.destroyAllWindows()
ser.close()
pool.release()
