import time
import mediapipe as mp

import numpy as np
from numpy import argmax
import PIL
import os
import matplotlib.pyplot as plt
import cv2
import sys
from keras.models import load_model
sys.path.append('./YOLOv7') # MUST put on import yolov7
from YOLOv7.yolov7_detector import initialize_model, detect_objects


weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"
model_CNN = load_model('./model_lenet_train_pose//lenet_epo20_bs5_at_home.h5')

categories = ['lie', 'sit', 'stand', 'minus']

WIDTH=128
HEIGHT=128
N_CHANNELS = 3
cap = cv2.VideoCapture("./converted_mp4_code/at_home/full_lie_wake_at_home.mp4")
model, device, half= initialize_model(weights_path)

FAST_FORWARD_RATE = 50
LIMIT_DISTANCE = 20
isSleep = []
f=0
while True:
    curr_full_time = time.time()
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + FAST_FORWARD_RATE)
    # curr = time.time()
    ret, img = cap.read()
    # Resize ảnh của toàn video
    if not ret:
        break
    if ret:
        height, width = img.shape[:2]
        curr = time.time()
        n = 5
        for i in range(n):
            if i == n - 1:
                width = int(width * 0.99)
                height = int(height * 0.99)
            else:
                width = int(width * 0.8)
                height = int(height * 0.8)
        resized_img = cv2.resize(img, (width, height))
        # resized_img = img
        now = time.time() - curr
        print('time rsize:  ', now)
    # Detect ảnh
    # curr = time.time()
    x, y, w, h, conf = detect_objects(model, device, resized_img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
    # now = time.time() - curr
    # print('time:  ', now)
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv2.putText(resized_img, str(conf) + "% YOLO", (x+60, y-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    # print("CONFFFFF:  ",str(conf))
    cv2.imshow('Display Image Roi', resized_img)
    
    # Resize ảnh của chủ thể người
    roi = resized_img[y:y+h,x:x+w]
    image_resize = cv2.resize(roi, (WIDTH, HEIGHT))
    print("shape1:  ", image_resize.shape)
    image_resize = image_resize.astype('float32')/255.0
    print("shape2:  ", image_resize.shape)

    image_expand = np.expand_dims(image_resize,axis=0)
    print("shape3:  ", image_expand.shape)
    pred=model_CNN.predict(image_expand)
    res = argmax(pred, axis=1)
    
    accuracy_CNN = pred[0][res][0] * 100
    accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
    print('categories[res]:   ',categories[res[0]])
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv2.putText(resized_img, str(accuracy_CNN_formatted) + "% CNN", (x+60, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(resized_img, categories[res[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.imshow('Image after predict pose', resized_img) 
    f+=1
    key=cv2.waitKey(20)
    now_full_time = time.time() - curr_full_time
    print('FULL TIME:  ', now_full_time)
    print('----------------------------------------------END-----------------------------------------------')
    if key ==ord('q'): 
        break 
cap.release()
cv2.destroyAllWindows()