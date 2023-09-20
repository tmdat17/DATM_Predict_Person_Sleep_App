import numpy as np
from numpy import argmax
import PIL
import matplotlib.pyplot as plt
import cv2
import sys
from keras.models import load_model
sys.path.append('./YOLOv7') # MUST put on import yolov7
from YOLOv7.yolov7_detector import initialize_model, detect_objects

WIDTH=128
HEIGHT=128
N_CHANNELS = 3
categories = ['lie', 'sit', 'stand']
weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"
model_CNN = load_model('./model_lenet_train_pose/lenet_epo20_bs5.h5')

img_path = './sample_img_pose/pose_lie_wi_2_resize_right.jpg'

img = cv2.imread(img_path)
model, device, half= initialize_model(weights_path)
x, y, w, h, conf = detect_objects(model, device, img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
x, y, w, h, conf = detect_objects(model, device, img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
print('TEST:    ', x,y,w,h)
cv2.rectangle(img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
cv2.imshow('Image with Bounding Box', img)
# cv2.waitKey(0)

roi = img[y:y+h,x:x+w]
image_resize = cv2.resize(roi, (WIDTH, HEIGHT))
print("shape1:  ", image_resize.shape)
image_resize = image_resize.astype('float32')/255.0
print("shape2:  ", image_resize.shape)

image_expand = np.expand_dims(image_resize,axis=0)
print("shape3:  ", image_expand.shape)
pred=model_CNN.predict(image_expand)
print('pred:   ',pred)
res = argmax(pred, axis=1)
accuracy_CNN = pred[0][res][0] * 100
accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
accuracy_CNN = pred[0][res][0] * 100
accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
print('categories[res]:   ',categories[res[0]])
cv2.rectangle(img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
cv2.putText(img, str(accuracy_CNN_formatted) + "%", (x + 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
cv2.putText(img, str(accuracy_CNN_formatted) + "%", (x + 80, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
cv2.putText(img, categories[res[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
cv2.imshow('Image after predict pose', img)
cv2.waitKey(0)