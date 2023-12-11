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

# model_extract = load_model('./model_extract_feature_CNN/lenet_ImgGray_extract_features_epo35_bs16_checked.h5')
# model_SVR = joblib.load('./models_lenet_SVR/Tuning_SVR/feature_lenet_SVR_epo35_bs16_grayImage_tuning.h5')

weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"
model_CNN = load_model('./model_lenet_train_pose/lenet_epo20_bs12_2500pics_new.h5')

categories = ['lie', 'sit', 'stand', 'minus']

WIDTH=128
HEIGHT=128
N_CHANNELS = 3
FAST_FORWARD_RATE = 50
LIMIT_DISTANCE = 8
EACH_FRAME = 5
isSleep = []
f=0

# cap = cv2.VideoCapture("./sample_img_pose/sleep.mp4")
# cap = cv2.VideoCapture("./Data_Train_Pose/lie_bonus.mov")
# cap = cv2.VideoCapture("./clip_cropped/full_lie_wake_at_home/20s/clip_15.mp4")
cap = cv2.VideoCapture('./VIDEO_TEST/video_demo.mp4')
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture("./Data_Train_Pose/sit_1p36.MOV")
# cap = cv2.VideoCapture("./Data_Train_Pose/stand_3p5.MOV")
# cap = cv2.VideoCapture("./Data_Train_Pose/stand_3p5.MOV")
model, device, half= initialize_model(weights_path)

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

lm_list = []

def make_landmark_timestep(results):
    # print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate (results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    coordinate = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id in fit_landmark :
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                coordinate.append([cx, cy])
                cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
    print('coordinate:  ', coordinate)
    return img, coordinate

# fit_landmark = [11, 12, 23, 24, 25, 26]
fit_landmark = [15, 16, 25, 26]
def euclidean_distance(point1, point2):
    point1 = np.array(point1)
    point2 = np.array(point2)
    squared_diff = (point1 - point2)**2
    squared_distance_sum = np.sum(squared_diff)
    distance = np.sqrt(squared_distance_sum)
    return distance

def euclidean_base_landmark(results, img, pre_point):
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            if id in fit_landmark :
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy)
                euclidean_distance
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
            
filename = 'lie_is_sleep_test.txt'
if not os.path.isfile(filename):
    open(filename, 'w').close()
else: open(filename, 'w').close()
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
    # cv2.imshow('Display Image Roi', resized_img)
    
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
    # if sit | lie => media pose
    if categories[res[0]] == 'lie' or categories[res[0]] == 'sit' or categories[res[0]] == 'stand':
        frameRGB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # ve khung xuong
            resized_img, coordinate = draw_landmark_on_image(mpDraw, results, resized_img)
            
        cv2.imshow('frame POSE camera' , resized_img)
    
        # Logic determinate sleeping
        if 'coordinate' in globals():
            cur_coordinate = coordinate
            if f % EACH_FRAME == 0:
                if 'pre_coordinate' in globals():
                    print('pre_coordinate:  ', pre_coordinate)
                    for i in range(len(coordinate)):
                        result_euclid = euclidean_distance(pre_coordinate[i], cur_coordinate[i])
                        print('{} result_euclid: {}'.format(f, result_euclid))
                        if result_euclid > LIMIT_DISTANCE:
                            with open(filename, 'a') as file:
                                if(i == len(coordinate) - 1):
                                    file.write('0\n')                                    
                                else: file.write('0 ')
                            isSleep.append(0)   # DONT SLEEP
                        elif result_euclid <= LIMIT_DISTANCE:
                            isSleep.append(1)   # SLEEP
                            with open(filename, 'a') as file:
                                if(i == len(coordinate) - 1):
                                    file.write('1\n')                                    
                                else: file.write('1 ')
                pre_coordinate = coordinate
        print('{} have isSleep: {}'.format(f, isSleep))
        
    # now = time.time() - curr
    # print('CNN time:  ', now)
    
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