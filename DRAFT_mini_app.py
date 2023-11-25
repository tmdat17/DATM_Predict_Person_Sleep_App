import time
import mediapipe as mp
import pandas as pd
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
import joblib

import tkinter as tk
from tkinter import Canvas, ttk
from tkinter import filedialog, IntVar, messagebox
import PIL.Image,PIL.ImageTk
# from PIL import Image, ImageTk
# ====================================================== Function Implement ======================================================

lm_list = []

def make_landmark_timestep(results):
    # print(results.pose_landmarks.landmark)
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
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
            if id in fit_landmark:
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(cx, cy)
                coordinate.append([cx, cy])
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
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
            if id in fit_landmark:
                h, w, c = img.shape
                print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(cx, cy)
                euclidean_distance
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)



def show_error_message(msg):
    messagebox.showerror(message=msg, title='Threshold value not empty!!')


weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"
model, device, half = initialize_model(weights_path)
model_CNN = load_model(
    './model_lenet_train_pose/lenet_epo20_bs12_2500pics_new.h5', compile=False)



categories = ['lie', 'sit', 'stand', 'minus']
mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

IS_PLAYING = True

EACH_FRAME = 5
LIMIT_DISTANCE = 2.5

WIDTH = 128
HEIGHT = 128

f = 0
STATUS = 'STATUS'
pre_coordinate = None
coordinate = None
isSleep = []
FAST_FORWARD_RATE = 25

file_path = None
selected_option = 3
selected_dropdown = 'SVM'

LIMIT_LINE = 3
LIMIT_FEATURE = LIMIT_LINE * 4

QUANTITY_1 = 0
def updateFilePath(file_path, selected_option, selected_dropdown, value_threshold):
    global QUANTITY_1
    global LIMIT_LINE
    global LIMIT_FEATURE
    global model_detect_sleep
    LIMIT_LINE = selected_option
    LIMIT_FEATURE = LIMIT_LINE * 4
    if value_threshold:
        THRESHOLD = int(value_threshold)
    model_detect_sleep = joblib.load(
        f'./model_detect_sleep/SVC/SVC_KFold_{LIMIT_LINE}_lines_K_50_C_100000_GAMMA_0.001.h5')
    if(file_path is not None):
        global cap
        cap = cv2.VideoCapture(file_path)
        if selected_dropdown == 'SVM':
            print('selected_option:  ', selected_option)
            print(f'\nLIMIT_LINE: {LIMIT_LINE}\nLIMIT_FEATURE: {LIMIT_FEATURE}')
            process_video_with_model_SVM()
        else:
            if not value_threshold:
                show_error_message('Please input Threshold value!!')
            else:
                QUANTITY_1 = LIMIT_LINE * 4 * THRESHOLD / 100
                print('QUANTITY_1:  ', QUANTITY_1)
                process_video_with_threshold()

# Hàm xử lý video with SVM
def process_video_with_model_SVM():
    global f
    global STATUS
    global pre_coordinate
    global coordinate
    global isSleep
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(
            cv2.CAP_PROP_POS_FRAMES) + FAST_FORWARD_RATE)
    ret, img = cap.read()
    # frame = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
    
    height, width = img.shape[:2]
    n = 5
    for i in range(n):
        if i == n - 1:
            width = int(width * 0.99)
            height = int(height * 0.99)
        else:
            width = int(width * 0.8)
            height = int(height * 0.8)
    resized_img = cv2.resize(img, (width, height))
    x, y, w, h, conf = detect_objects(model, device, resized_img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
    # now = time.time() - curr
    # print('time:  ', now)
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv2.putText(resized_img, str(conf) + "% YOLO", (x+60, y-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    # cv2.imshow(f'Result', resized_img)
    # Resize ảnh của chủ thể người
    roi = resized_img[y:y+h,x:x+w]
    image_resize = cv2.resize(roi, (WIDTH, HEIGHT))
    image_resize = image_resize.astype('float32')/255.0
    image_expand = np.expand_dims(image_resize,axis=0)
    pred=model_CNN.predict(image_expand)
    res = argmax(pred, axis=1)
    accuracy_CNN = pred[0][res][0] * 100
    accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
    
    result_name_pose.config(text = categories[res[0]])
    result_accuracy_pose.config(text = accuracy_CNN_formatted)
    
    if categories[res[0]] == 'lie':
        frameRGB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # ve khung xuong
            # global coordinate
            resized_img, coordinate = draw_landmark_on_image(mpDraw, results, resized_img)
        if coordinate is not None:
            cur_coordinate = coordinate
            if f % EACH_FRAME == 0:
                if pre_coordinate is not None:
                    print('pre_coordinate:  ', pre_coordinate)
                    print('cur_coordinate:  ', cur_coordinate)
                    for i in range(len(coordinate)):
                        result_euclid = euclidean_distance(pre_coordinate[i], cur_coordinate[i])
                        print('{} result_euclid: {}'.format(f, result_euclid))
                        if result_euclid > LIMIT_DISTANCE:
                            global isSleep
                            isSleep.append(0)   # DONT SLEEP
                        elif result_euclid <= LIMIT_DISTANCE:
                            isSleep.append(1)   # SLEEP
                # global pre_coordinate
                pre_coordinate = coordinate
                print('isSLeep:  ', isSleep)
        print('{} have isSleep: {}\nsize: {}'.format(f, isSleep, len(isSleep)))
        if(len(isSleep) == LIMIT_FEATURE):
            test_df = pd.DataFrame([isSleep])
            preds = model_detect_sleep.predict(test_df)
            if(preds[0] == 's'):
                print('preds:  SLEEP', )
                STATUS = 'SLEEP'
                result_status_label.config(text = STATUS)
            else: 
                print('preds:  WAKE', )
                STATUS = 'WAKE'
                result_status_label.config(text = STATUS)
            isSleep = []
    accuracy_CNN = pred[0][res][0] * 100
    accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
    print('categories[res]:   ',categories[res[0]])
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv2.putText(resized_img, str(accuracy_CNN_formatted) + "% CNN", (x+60, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(resized_img, categories[res[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(resized_img, STATUS, (x , y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    
    f+=1
    img_convert_color = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGBA)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_convert_color))
    # cv2.imshow(f'Result', resized_img)
    result_img.imgtk = photo
    result_img.configure(image=photo, height=675)
    if IS_PLAYING:
        result_img.after(15, process_video_with_model_SVM)
    else: print('----------------------------------------------------------------------------------')
    
# Hàm xử lý video with Threshold
def process_video_with_threshold():
    global f
    global STATUS
    global pre_coordinate
    global coordinate
    global isSleep
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(
            cv2.CAP_PROP_POS_FRAMES) + FAST_FORWARD_RATE)
    ret, img = cap.read()
    # frame = cv2.resize(img, dsize=None, fx=0.5, fy=0.5)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to RGB
    
    height, width = img.shape[:2]
    n = 5
    for i in range(n):
        if i == n - 1:
            width = int(width * 0.99)
            height = int(height * 0.99)
        else:
            width = int(width * 0.8)
            height = int(height * 0.8)
    resized_img = cv2.resize(img, (width, height))
    x, y, w, h, conf = detect_objects(model, device, resized_img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
    # now = time.time() - curr
    # print('time:  ', now)
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv2.putText(resized_img, str(conf) + "% YOLO", (x+60, y-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    # cv2.imshow(f'Result', resized_img)
    # Resize ảnh của chủ thể người
    roi = resized_img[y:y+h,x:x+w]
    image_resize = cv2.resize(roi, (WIDTH, HEIGHT))
    image_resize = image_resize.astype('float32')/255.0
    image_expand = np.expand_dims(image_resize,axis=0)
    pred=model_CNN.predict(image_expand)
    res = argmax(pred, axis=1)
    accuracy_CNN = pred[0][res][0] * 100
    accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
    
    result_name_pose.config(text = categories[res[0]])
    result_accuracy_pose.config(text = accuracy_CNN_formatted)
    
    if categories[res[0]] == 'lie':
        frameRGB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # ve khung xuong
            # global coordinate
            resized_img, coordinate = draw_landmark_on_image(mpDraw, results, resized_img)
        if coordinate is not None:
            cur_coordinate = coordinate
            if f % EACH_FRAME == 0:
                if pre_coordinate is not None:
                    print('pre_coordinate:  ', pre_coordinate)
                    print('cur_coordinate:  ', cur_coordinate)
                    for i in range(len(coordinate)):
                        result_euclid = euclidean_distance(pre_coordinate[i], cur_coordinate[i])
                        print('{} result_euclid: {}'.format(f, result_euclid))
                        if result_euclid > LIMIT_DISTANCE:
                            global isSleep
                            isSleep.append(0)   # DONT SLEEP
                        elif result_euclid <= LIMIT_DISTANCE:
                            isSleep.append(1)   # SLEEP
                # global pre_coordinate
                pre_coordinate = coordinate
                print('isSLeep:  ', isSleep)
        print('{} have isSleep: {}\nsize: {}'.format(f, isSleep, len(isSleep)))
        print(f'\ncount_1 current: {isSleep.count(1)}\n')
        if(len(isSleep) == LIMIT_FEATURE):
            count_ones = isSleep.count(1)
            if(count_ones >= QUANTITY_1):
                print('preds:  SLEEP', )
                STATUS = 'SLEEP'
                result_status_label.config(text = STATUS)
            else: 
                print('preds:  WAKE', )
                STATUS = 'WAKE'
                result_status_label.config(text = STATUS)
            isSleep = []
    accuracy_CNN = pred[0][res][0] * 100
    accuracy_CNN_formatted = "{:.2f}".format(accuracy_CNN)
    print('categories[res]:   ',categories[res[0]])
    cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
    cv2.putText(resized_img, str(accuracy_CNN_formatted) + "% CNN", (x+60, y-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(resized_img, categories[res[0]], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    cv2.putText(resized_img, STATUS, (x , y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
    
    f+=1
    img_convert_color = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGBA)
    photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(img_convert_color))
    # cv2.imshow(f'Result', resized_img)
    result_img.imgtk = photo
    result_img.configure(image=photo, height=675)
    if IS_PLAYING:
        result_img.after(15, process_video_with_threshold)
    else: print('----------------------------------------------------------------------------------')



# Hàm mở hộp thoại để chọn tệp video cũng như submit tất cả các option
def choose_file():
    global STATUS
    global pre_coordinate
    global coordinate
    global f
    global isSleep
    global IS_PLAYING
    global file_path
    global selected_option
    global selected_dropdown
    file_path = filedialog.askopenfilename(
    filetypes=[("Video files", "*.mp4")])
    selected_option = selected.get()
    selected_dropdown = valueDropdown.get()
    # value_threshold = int(input_threshold.get())
    value_threshold = input_threshold.get()
    if(value_threshold):
        if('%' in value_threshold):
            value_threshold = int(str(value_threshold).strip().split('%')[0])
        else: int(value_threshold)
    print('value_option:  ',selected_option)
    print('value_dropdown:  ',selected_dropdown)
    print('value_threshold:  ',value_threshold)
    if file_path:
        IS_PLAYING = True
        isSleep = []
        f = 0
        STATUS = 'STATUS'
        pre_coordinate = None
        coordinate = None
        updateFilePath(file_path, selected_option, selected_dropdown, value_threshold)


def stopProcess():
    global IS_PLAYING
    IS_PLAYING = False

def clearResult():
    result_name_pose.config(text='none')
    result_accuracy_pose.config(text='none')
    result_status_label.config(text='none')


def quitApp():
    global stop_threads
    stop_threads = True
    app.destroy()


def update_entry_state(event):
    selected_dropdown = valueDropdown.get()
    if selected_dropdown == 'Threshold':
        input_threshold.config(state=tk.NORMAL)
    else:
        input_threshold.config(state=tk.DISABLED)



# ====================================================== UI ======================================================

# Tạo cửa sổ ứng dụng Tkinter
app = tk.Tk()
app.geometry('1000x780+450+5')

app.title("Mini App Detect Person Sleep")

# Create a label
label = tk.Label(app, 
                 text="Mini App Detect Person Sleep!",
                 font=("Times New Roman", 23, 'bold')).place(x = 390, y = 5)

# Create radio buttons

selected = tk.IntVar()
selected.set(3)
option1 = tk.Radiobutton(app, 
                         text="12 features (3 lines)", 
                         font=("Times New Roman", 13), 
                         variable=selected, 
                         value=3, 
                         activebackground='red',
                        )

option2 = tk.Radiobutton(app, 
                         text="16 features (4 lines)", 
                         font=("Times New Roman", 13), 
                         variable=selected, 
                         value=4, 
                         activebackground='red'
                        )

option3 = tk.Radiobutton(app,
                         text="20 features (5 lines)",
                         font=("Times New Roman", 13),
                         variable=selected,
                         value=5,
                         activebackground='red'
                        )

option1.place(x=20, y=130)
option2.place(x=20, y=170)
option3.place(x=20, y=210)





# create dropdown
valueDropdown = tk.StringVar() 
optionModel = ttk.Combobox(app, 
                           width = 20, 
                           textvariable = valueDropdown, 
                           font=("Times New Roman", 13),
                           ) 
  
# Adding combobox drop down list 
optionModel['values'] = ('SVM',  
                          'Threshold', )
optionModel.place(x = 20, y = 265)
optionModel.current(0)
optionModel.bind("<<ComboboxSelected>>", update_entry_state)


# create box input threshold
threshold_label = tk.Label(app, 
                        text="Threshold:  ",
                        font = ("Times New Roman", 15),
                        )
threshold_label.place(x=20, y=307)

input_threshold = tk.Entry(app, 
                           state=tk.DISABLED, 
                           relief='groove', 
                           justify='center',
                           font = ("Times New Roman", 13),
                           width= '10',
                           highlightcolor='yellow'
                           )
input_threshold.place(x=120, y=310)


# Create button choose file video
open_file_icon=PIL.Image.open("./icon_app/open_file.png")
open_file_icon_resize=open_file_icon.resize((25,25),PIL.Image.ANTIALIAS)
open_file_icon_img=PIL.ImageTk.PhotoImage(open_file_icon_resize)
choose_file_button = tk.Button(app,
                               text="Choose Video ",
                               font = ("Times New Roman", 11),
                               width="115",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=choose_file,
                               image=open_file_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white'
                            ).place(x = 20, y = 370)

camera_icon=PIL.Image.open("./icon_app/open_camera.png")
camera_icon_resize=camera_icon.resize((25,25),PIL.Image.ANTIALIAS)
camera_icon_img=PIL.ImageTk.PhotoImage(camera_icon_resize)
choose_file_button = tk.Button(app,
                               text="Use Camera ",
                               font = ("Times New Roman", 11),
                               width="112",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=choose_file,
                               image=camera_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white').place(x = 160, y = 370)


stop_icon=PIL.Image.open("./icon_app/stop.png")
stop_icon_resize=stop_icon.resize((25,25),PIL.Image.ANTIALIAS)
stop_icon_img=PIL.ImageTk.PhotoImage(stop_icon_resize)
choose_file_button = tk.Button(app,
                               text="STOP PROCESS     ",
                               font = ("Times New Roman", 11),
                               width="250",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=stopProcess,
                               image=stop_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white'
                            ).place(x = 20, y = 440)


clean_result_icon=PIL.Image.open("./icon_app/clean_result.png")
clean_result_icon_resize=clean_result_icon.resize((25,25),PIL.Image.ANTIALIAS)
clean_result_icon_img=PIL.ImageTk.PhotoImage(clean_result_icon_resize)
choose_file_button = tk.Button(app,
                               text="CLEAN RESULT     ",
                               font = ("Times New Roman", 11),
                               width="250",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=clearResult,
                               image=clean_result_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white'
                            ).place(x = 20, y = 500)


quit_app_icon=PIL.Image.open("./icon_app/exit_app.png")
quit_app_icon_resize=quit_app_icon.resize((25,25),PIL.Image.ANTIALIAS)
quit_app_icon_img=PIL.ImageTk.PhotoImage(quit_app_icon_resize)
choose_file_button = tk.Button(app,
                               text="QUIT   ",
                               font = ("Times New Roman", 11),
                               width="250",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=quitApp,
                               image=quit_app_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white'
                            ).place(x = 20, y = 560)


# Đường phân chia
line_separate = tk.Canvas(app, width=20, height=800)
line_separate.place(x=300, y=60)
line_separate.create_line((10, 15), (10, 650), width=1, fill='gray')


# Tạo một `Canvas` cho cột phải để hiển thị kết quả
result_board = tk.Canvas(app, width=630, height=680, bg='white', borderwidth=8, relief="groove")
result_board.place(x=330, y=50)


# khung của video hiển thị lên tkinter
result_img = tk.Label(result_board, background='white')
result_img.place(x=305, y=10)



# create title result board
name_pose_label = tk.Label(app, 
                        text="RESULT ",
                        font = ("Times New Roman", 25, 'bold'),
                        fg="black",
                        bg="white"
                        )
name_pose_label.place(x=420, y=85)



# create label for name CNN predict
name_pose_label = tk.Label(app, 
                        text="The pose is: ",
                        font = ("Times New Roman", 15, 'bold'),
                        fg="black",
                        bg="white"
                        )
name_pose_label.place(x=390, y=200)

result_name_pose = tk.Label(app, 
                        text="none",
                        font = ("Times New Roman", 18, 'bold'),
                        fg="red",
                        bg="white"
                        )
result_name_pose.place(x=550, y=200)

accuracy_pose_label = tk.Label(app, 
                        text="Accuracy is: ",
                        font = ("Times New Roman", 15, 'bold'),
                        fg="black",
                        bg="white"
                        )
accuracy_pose_label.place(x=390, y=260)

result_accuracy_pose = tk.Label(app, 
                        text="none",
                        font = ("Times New Roman", 18, 'bold'),
                        fg="red",
                        bg="white"
                        )
result_accuracy_pose.place(x=550, y=260)



# create label for name predict isSleep
name_status_label = tk.Label(app, 
                        text="Status is: ",
                        font = ("Times New Roman", 15, 'bold'),
                        fg="black",
                        bg="white"
                        )
name_status_label.place(x=390, y=320)

result_status_label = tk.Label(app, 
                        text="none",
                        font = ("Times New Roman", 18, 'bold'),
                        fg="red",
                        bg="white"
                        )
result_status_label.place(x=550, y=320)


# Tạo phần nền màu đen
footer = tk.Canvas(app, width=1005, height=27, bg="#22272e")
footer.place(x=-5, y=753)

# create footer
footer_label = tk.Label(app, 
                        text="datb1913221@student.ctu.edu.vn",
                        font = ("Times New Roman", 12, 'italic'),
                        fg="white",
                        bg="#22272e")
footer_label.place(x=500, y=767, anchor="center")

# app.resizable(False,False)
app.mainloop()