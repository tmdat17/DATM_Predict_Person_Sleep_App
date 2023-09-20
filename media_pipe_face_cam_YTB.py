import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

lm_list = []

def draw_landmarks_on_image(results):
    for face_landmarks in results.multi_face_landmarks:
        # mp_drawing.draw_landmarks(
        #     image = img,
        #     landmark_list = face_landmarks,
        #     connections = mp_face_mesh.FACEMESH_IRISES,
        #     landmark_drawing_spec = None,
        #     connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        # )
        mp_drawing.draw_landmarks(
            image = img,
            landmark_list = face_landmarks,
            connections = mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec = None,
            connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style()
        )
        # mp_drawing.draw_landmarks(
        #     image = img,
        #     landmark_list = face_landmarks,
        #     connections = mp_face_mesh.FACEMESH_TESSELATION,
        #     landmark_drawing_spec = None,
        #     connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style()
        # )


def make_landmark_timestep(results):
    c_lm = []
    coordinate_list = []
    for id, lm in enumerate (results.multi_face_landmarks[0].landmark):
        # 0.5757957696914673
        coordinate = {
            'x': lm.x,
            'y': lm.y,
            'z': lm.z,
        }
        coordinate_list.append(coordinate)
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
    print('len coordinate_list:  ', len(coordinate_list))
    print('coordinate_list:   ', coordinate_list)
    # print('coordinate_list {}:  {}  '.format(0, coordinate_list[0]))
    
    return c_lm

# cap = cv2.VideoCapture('192.168.1.34:4747')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # apply model FACE MESH with mediapipe
    if ret:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.FaceMesh(refine_landmarks=True).process(img)
        # print('results:  ', results.multi_face_landmarks[0])
        
        # draw annotations on the image

        # check the face is existed
        if results.multi_face_landmarks:
            lm = make_landmark_timestep(results)
            lm_list.append(lm)
            draw_landmarks_on_image(results)

        cv2.imshow('frame FACE MESH camera', cv2.flip(img,1))
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()