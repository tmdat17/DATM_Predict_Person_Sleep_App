import cv2
import os
from yolov7_detector import initialize_model, detect_objects


# cap = cv2.VideoCapture("192.168.1.10:4747")
weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"


# --------------------------------------------- Static Image
# img_path = 'E:\CTU\LUAN_VAN_2023\sample_img_pose\pose_sit_wi_1_resize.jpg'

# img = cv2.imread(img_path)
# model, device, half= initialize_model(weights_path)
# x, y, w, h = detect_objects(model, device, img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
# # x, y, w, h = detect_objects(model, device, img_path, half, img_size=640, conf_thres=0.1, iou_thres=0.45)
# print('TEST:    ', x,y,w,h)
# cv2.rectangle(img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
# cv2.imshow('Image with Bounding Box', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# --------------------------------------------- Video
# HEIGHT = 128
# WIDTH = 128
index = 0
# action = 'stand'
# cap = cv2.VideoCapture('../Data_Train_Pose/full_{}.mov'.format(action))
cap = cv2.VideoCapture('../converted_mp4_code/at_home/full_lie_sleep_at_home_v2.mp4')
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

        # Resize ảnh
        resized_img = cv2.resize(frame, (width, height))

        x, y, w, h, confidence_formatted = detect_objects(model, device, resized_img, half, img_size=640, conf_thres=0.1, iou_thres=0.45)

        roi = resized_img[y:y+h,x:x+w]
        
        # if roi is not None and roi.size != 0:
        #     print("prepare excute image {}:".format(index))
        #     # roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        #     # roi_gray = cv2.resize(roi_gray,(WIDTH,HEIGHT),interpolation=cv2.INTER_AREA)
        #     if not os.path.exists('../data_cropped/'):
        #         os.makedirs('../data_cropped/')
        #     if action == 'stand':
        #         if not os.path.exists('../data_cropped/stand'):
        #             os.makedirs('../data_cropped/stand')
        #         name_img = '{}_{}.jpg'.format(action, index)
        #         filename="../data_cropped/{}/".format(action)+str(name_img)
        #         # cv2.imwrite(filename,roi_gray)
        #         cv2.imwrite(filename,roi)
        #         print("saved image {}:".format(index))
        #         print("path: {}".format(filename))
        #         index += 1
        #     elif action == 'sit':
        #         if not os.path.exists('../data_cropped/sit'):
        #             os.makedirs('./data_cropped/sit')
        #         name_img = '{}_{}.jpg'.format(action, index)
        #         filename="../data_cropped/{}/".format(action)+str(name_img)
        #         # cv2.imwrite(filename,roi_gray)
        #         cv2.imwrite(filename,roi)
        #         print("saved image {}:".format(index))
        #         index += 1
        #     else:
        #         if not os.path.exists('../data_cropped/lie'):
        #             os.makedirs('../data_cropped/lie')
        #         name_img = '{}_{}.jpg'.format(action, index)
        #         filename="../data_cropped/{}/".format(action)+str(name_img)
        #         # cv2.imwrite(filename,roi_gray)
        #         cv2.imwrite(filename,roi)
        #         print("saved image {}:".format(index))
        #         index += 1
        # else:
        #     print('Error: Invalid image!')
        
        cv2.rectangle(resized_img, (x, y), (x+w, y+h), color = (0, 0, 255), thickness=2)
        cv2.putText(resized_img, str(confidence_formatted) + "% YOLO", (x+60, y-90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2, cv2.LINE_AA)
        cv2.imshow('Video', resized_img)

        # Bấm 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()