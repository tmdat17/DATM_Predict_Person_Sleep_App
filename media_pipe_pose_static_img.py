import cv2
import mediapipe as mp

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def make_landmark_timestep(results):
    # print('result.pose_landmarks:   ',results.pose_landmarks.landmark)
    c_lm = []
    coordinate_list = []
    for id, lm in enumerate (results.pose_landmarks.landmark):
        coordinate = {
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
            'visibility': lm.visibility,
        }
        coordinate_list.append(coordinate)
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    print('len coordinate_list pose:  ', len(coordinate_list))
    print('coordinate_list pose {} {}:  '.format(3, coordinate_list[3]))
    return c_lm

def draw_landmark_on_image(mpDraw, results, img):
    # match point
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    # draw each point
    for id, lm in enumerate(results.pose_landmarks.landmark):
        h, w, c = img.shape
        print(id, lm)
        # cx, cy = int(lm.x + w), int(lm.y + h)
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (cx, cy), 3, (0, 0, 0), cv2.FILLED)
    cv2.imshow('function:  ', img)
    cv2.waitKey(0)
    return img

# img = cv2.imread('./sample_img_pose/pose_lie.png')
# img = cv2.imread('./sample_img_pose/pose_sit.jpg')
# img = cv2.imread('./sample_img_pose/pose_stand.jpg')
img = cv2.imread('./sample_img_pose/RIFM.png')
# img = cv2.imread('./sample_img_pose/pose_lie_wi_2_resize.jpg')
# img = cv2.imread('./sample_img_pose/pose_lie_wi_3_resize.jpg')
cv2.imshow('static img', img)
frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(frameRGB)
if results.pose_landmarks:
    # ghi nhan thong so khung xuong
    lm = make_landmark_timestep(results)
    img_processed = draw_landmark_on_image(mpDraw, results, img)
    cv2.imshow('img processed', img_processed)


cv2.waitKey(0)
