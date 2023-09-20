import cv2
import mediapipe as mp

# cap = cv2.VideoCapture('./sample_img_pose/sleep.mp4')
cap = cv2.VideoCapture('./Data_Train_Pose/at_home/full_lie_sleep_at_home.MOV')

# cap = cv2.VideoCapture('./Data_Train_Pose/lie_bonus.mov')
# cap = cv2.VideoCapture('./Data_Train_Pose/full_lie.mov')

# init mediapipe
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
# fit_landmark = [11, 12, 23, 24, 25, 26]
fit_landmark = [9, 10, 11, 12, 23, 24, 25, 26]
def draw_landmark_on_image(mpDraw, results, img):
    # mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    for id, lm in enumerate(results.pose_landmarks.landmark):
        if id in fit_landmark :
            h, w, c = img.shape
            # print(id, lm)
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 3, (255, 0, 0), cv2.FILLED)
    return img

f=0
while True:
    ret, frame = cap.read()
    if ret:
        # resize img
        height, width = frame.shape[:2]
        print('f: {}'.format(f))
        n = 5
        for i in range(n):
            if i == n - 1:
                width = int(width * 0.99)
                height = int(height * 0.99)
            else:
                width = int(width * 0.8)
                height = int(height * 0.8)
        resized_img = cv2.resize(frame, (width, height))
        
        # recognize Pose
        frameRGB = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            # ghi nhan thong so khung xuong
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            # ve khung xuong
            resized_img = draw_landmark_on_image(mpDraw, results, resized_img)
        cv2.imshow('frame POSE camera' , resized_img)
        f+=1
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()