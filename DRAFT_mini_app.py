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


# Hàm xử lý video
def process_video(video_path, selected_option):
    LIMIT_LINE = selected_option
    print('LIMIT_LINE:     ', LIMIT_LINE)
    weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"
    model_CNN = load_model(
        './model_lenet_train_pose/lenet_epo20_bs12_2500pics_new.h5', compile=False)
    model_detect_sleep = joblib.load(
        f'./model_detect_sleep/SVC/SVC_KFold_{LIMIT_LINE}_lines_K_50_C_100000_GAMMA_0.001.h5')
    categories = ['lie', 'sit', 'stand', 'minus']

    WIDTH = 128
    HEIGHT = 128
    N_CHANNELS = 3
    # Đây là nơi bạn nên đưa mã nguồn của mô hình của mình

    # Ví dụ: Sử dụng OpenCV để đọc video
    cap = cv2.VideoCapture(video_path)
    model, device, half = initialize_model(weights_path)

    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    mpDraw = mp.solutions.drawing_utils

    FAST_FORWARD_RATE = 25
    LIMIT_DISTANCE = 2.5
    EACH_FRAME = 5
    isSleep = []
    f = 0
    coordinate = None
    pre_coordinate = None
    LIMIT_FEATURE = LIMIT_LINE * 4
    STATUS = 'STATUS'
    while True:
        print('IS SLEEPPPPP:     ', isSleep)
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(
            cv2.CAP_PROP_POS_FRAMES) + FAST_FORWARD_RATE)
        ret, img = cap.read()

        if not ret:
            break
        if ret:
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
            # resized_img = img
        # Ví dụ: Hiển thị frame sau xử lý
        x, y, w, h, conf = detect_objects(model, device, resized_img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
        # now = time.time() - curr
        # print('time:  ', now)
        cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
        cv2.putText(resized_img, str(conf) + "% YOLO", (x+60, y-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
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
                                isSleep.append(0)   # DONT SLEEP
                            elif result_euclid <= LIMIT_DISTANCE:
                                isSleep.append(1)   # SLEEP
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
        cv2.imshow(f'Result', resized_img) 
        f+=1
        key=cv2.waitKey(20)
        if key ==ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm mở hộp thoại để chọn tệp video


def process_option(option):
    print('option:  ', option)
    global selected_option
    selected_option = option

def show_error_message(msg):
    messagebox.showerror(msg)

def choose_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4")])
    if file_path:
        if selected_option == 3 or selected_option == 4 or selected_option == 5:
            process_video(file_path, selected_option)
        else: show_error_message('Select option not empty!')
        
# Tạo cửa sổ ứng dụng Tkinter
app = tk.Tk()
app.geometry('1000x750+150+20')

app.title("Mini App Detect Person Sleep")

# Create a label
label = tk.Label(app, text="Mini App Detect Person Sleep!",font=("Arial Bold", 20)).place(x = 400, y = 20)

# Create radio buttons
selected_option = tk.IntVar()
selected_option.set(3)
option1 = tk.Radiobutton(app, 
                         text="12 features (3 lines)", 
                         font=("Arial Bold", 10), 
                         variable=selected_option, 
                         value=3, 
                        #  command=lambda: process_option(3),
                         activebackground='red',
                        )

option2 = tk.Radiobutton(app, 
                         text="16 features (4 lines)", 
                         font=("Arial Bold", 10), 
                         variable=selected_option, 
                         value=4, 
                        #  command=lambda: process_option(4),
                         activebackground='red'
                        )

option3 = tk.Radiobutton(app,
                         text="20 features (5 lines)",
                         font=("Arial Bold", 10),
                         variable=selected_option,
                         value=5,
                        #  command=lambda: process_option(5),
                         activebackground='red'
                        )

option1.place(x=80, y=120)
option2.place(x=80, y=160)
option3.place(x=80, y=200)





# create cropdown
n = tk.StringVar() 
optionModel = ttk.Combobox(app, width = 20, textvariable = n) 
  
# Adding combobox drop down list 
optionModel['values'] = (' SVM',  
                          ' Median', 
                          ' Mean', ) 
  
optionModel.place(x = 80, y = 230)
optionModel.current(0) 

# Create button choose file video
open_file_icon=PIL.Image.open("./icon_app/open_file.png")
open_file_icon_resize=open_file_icon.resize((25,25),PIL.Image.ANTIALIAS)
open_file_icon_img=PIL.ImageTk.PhotoImage(open_file_icon_resize)
choose_file_button = tk.Button(app,
                               text="Choose Video ",
                               width="105",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=choose_file,
                               image=open_file_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white'
                            ).place(x = 30, y = 280)

camera_icon=PIL.Image.open("./icon_app/open_camera.png")
camera_icon_resize=camera_icon.resize((25,25),PIL.Image.ANTIALIAS)
camera_icon_img=PIL.ImageTk.PhotoImage(camera_icon_resize)
choose_file_button = tk.Button(app,
                               text="Use Camera ",
                               width="100",
                               height="30",
                               borderwidth=4,
                               relief="ridge",
                               command=choose_file,
                               image=camera_icon_img,
                               compound = "right",
                               activebackground='#a9b1b6',
                               activeforeground='white').place(x = 180, y = 280)


# choose_camera_button = tk.Button(app, text="Use Camera", command=choose_file).place(x = 120, y = 280)

# Đường phân chia
line_separate = tk.Canvas(app, width=20, height=800)
line_separate.place(x=300, y=60)
line_separate.create_line((10, 15), (10, 650), width=1, fill='gray')


# Tạo một `Canvas` cho cột phải để hiển thị kết quả
result_board = tk.Canvas(app, width=630, height=620, bg='white', borderwidth=8, relief="groove")
result_board.place(x=330, y=80)

# author=tk.Label(app, text = "datb1913221@student.ctu.edu.vn",
#           font = ("Times New Roman", 12),
#           fg='white',
#           bg="black",
#         )
# author.place(x=0, y=730)

# Tạo phần nền màu đen
footer = tk.Canvas(app, width=1005, height=27, bg="#22272e")
footer.place(x=-5, y=723)

# create footer
footer_label = tk.Label(app, 
                        text="datb1913221@student.ctu.edu.vn",
                        font = ("Times New Roman", 12, 'italic'),
                        fg="white",
                        bg="#22272e")
footer_label.place(x=500, y=737, anchor="center")


# create label for name CNN predict
name_pose_label = tk.Label(app, 
                        text="The pose is: ",
                        font = ("Times New Roman", 15, 'bold'),
                        fg="black",
                        bg="white"
                        )
name_pose_label.place(x=440, y=180)

result_name_pose = tk.Label(app, 
                        text="none",
                        font = ("Times New Roman", 18, 'bold'),
                        fg="red",
                        bg="white"
                        )
result_name_pose.place(x=600, y=180)

accuracy_pose_label = tk.Label(app, 
                        text="Accuracy is: ",
                        font = ("Times New Roman", 15, 'bold'),
                        fg="black",
                        bg="white"
                        )
accuracy_pose_label.place(x=440, y=240)

result_accuracy_pose = tk.Label(app, 
                        text="none",
                        font = ("Times New Roman", 18, 'bold'),
                        fg="red",
                        bg="white"
                        )
result_accuracy_pose.place(x=600, y=240)



# create label for name predict isSleep
name_status_label = tk.Label(app, 
                        text="The person is: ",
                        font = ("Times New Roman", 15, 'bold'),
                        fg="black",
                        bg="white"
                        )
name_status_label.place(x=440, y=300)

result_status_label = tk.Label(app, 
                        text="none",
                        font = ("Times New Roman", 18, 'bold'),
                        fg="red",
                        bg="white"
                        )
result_status_label.place(x=600, y=300)

# accuracy_status_label = tk.Label(app, 
#                         text="Accuracy is: ",
#                         font = ("Times New Roman", 15, 'bold'),
#                         fg="black",
#                         bg="white"
#                         )
# accuracy_status_label.place(x=440, y=360)

# result_accuracy_status = tk.Label(app, 
#                         text="none",
#                         font = ("Times New Roman", 15, 'bold'),
#                         fg="red",
#                         bg="white"
#                         )
# result_accuracy_status.place(x=600, y=360)


app.resizable(False,False)
app.mainloop()