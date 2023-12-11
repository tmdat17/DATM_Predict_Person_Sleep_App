import cv2
import os
import sys
sys.path.append('./YOLOv7') 
from YOLOv7.yolov7_detector import initialize_model, detect_objects


# cap = cv2.VideoCapture("192.168.1.10:4747")
weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"


# --------------------------------------------- Static Image (use when test model static img, camera, video)
# img_path = 'E:\CTU\LUAN_VAN_2023\sample_img_pose\pose_lie_wi_1_resize.jpg'

# img = cv2.imread(img_path)
# model, device, half= initialize_model(weights_path)
# x, y, w, h, confidence_formatted = detect_objects(model, device, img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
# print('TEST:    ', x,y,w,h)
# cv2.rectangle(img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
# cv2.imshow('Image with Bounding Box', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --------------------------------------------- Video (use when crop img from video (has data crop img) for train model)
HEIGHT = 128
WIDTH = 128
index = 1351
LIMIT_IMG = 2500
action = 'sit_sleep'
folder_save = 'sit'
file_name = 'full_{}_at_home'.format(action)
cap = cv2.VideoCapture('./converted_mp4_code/at_home/{}.mp4'.format(file_name))
# cap = cv2.VideoCapture('./Data_Train_Pose/full_{}.mov'.format(action))
model, device, half= initialize_model(weights_path)
fast_forward_rate = 15
while True:
    # Đặt vị trí hiện tại của video để tua nhanh
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + fast_forward_rate)
    ret, frame = cap.read()

    if ret:
        height, width = frame.shape[:2]
        for i in range(5):
            if i==4:
                width = int(width * 0.99)
                height = int(height * 0.99)
            else:
                width = int(width * 0.8)
                height = int(height * 0.8)

    #     # Resize ảnh
        resized_img = cv2.resize(frame, (width, height))

        x, y, w, h, confidence_formatted = detect_objects(model, device, resized_img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)

        roi = resized_img[y:y+h,x:x+w]
        
        if roi is not None and roi.size != 0 and index < LIMIT_IMG:
            print("prepare excute image {}:".format(index))
            # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # roi_gray = cv2.resize(roi_gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
            if not os.path.exists('./data_cropped/at_home'):
                os.makedirs('./data_cropped/at_home')
            if not os.path.exists('./data_cropped/at_home/{}/'.format(folder_save)):
                os.makedirs('./data_cropped/at_home/{}/'.format(folder_save))
            name_img = '{}_{}.jpg'.format(action, index)
            filename="./data_cropped/at_home/{}/".format(folder_save)+str(name_img)
            # cv2.imwrite(filename,roi_gray)
            cv2.imwrite(filename,roi)
            print("saved image {}:".format(index))
            print("path: {}".format(filename))
            index += 1
            # if action == 'stand':
            #     if not os.path.exists('./data_cropped/stand'):
            #         os.makedirs('./data_cropped/stand')
            #     name_img = '{}_{}.jpg'.format(action, index)
            #     filename="./data_cropped/{}/".format(action)+str(name_img)
            #     # cv2.imwrite(filename,roi_gray)
            #     cv2.imwrite(filename,roi)
            #     print("saved image {}:".format(index))
            #     print("path: {}".format(filename))
            #     index += 1
            # elif action == 'sit':
            #     if not os.path.exists('./data_cropped/sit'):
            #         os.makedirs('./data_cropped/sit')
            #     name_img = '{}_{}.jpg'.format(action, index)
            #     filename="./data_cropped/{}/".format(action)+str(name_img)
            #     # cv2.imwrite(filename,roi_gray)
            #     cv2.imwrite(filename,roi)
            #     print("saved image {}:".format(index))
            #     print("path: {}".format(filename))
            #     index += 1
            # else:
            #     if not os.path.exists('./data_cropped/lie'):
            #         os.makedirs('./data_cropped/lie')
            #     name_img = '{}_{}.jpg'.format(action, index)
            #     filename="./data_cropped/{}/".format(action)+str(name_img)
            #     # cv2.imwrite(filename,roi_gray)
            #     cv2.imwrite(filename,roi)
            #     print("saved image {}:".format(index))
            #     print("path: {}".format(filename))
            #     index += 1
        else:
            if(index >= LIMIT_IMG):
                print('LIMIT IMAGE!!!')
                break
            else: print('Error: Invalid image!') 
            
        
        # cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
        # cv2.imshow('Video', resized_img)

        # Bấm 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()