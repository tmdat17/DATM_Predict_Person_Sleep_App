import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import functools
import time

#import model is trained
model_path = './model_trained_face_landmark_detection/face_landmarker.task'

# create the task
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()

# Create a face landmarker instance with the live stream mode:
# output_image = annotated_image (static_img file)


def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    current_time = time.time()  # Thời gian hiện tại
    elapsed_time = current_time - (timestamp_ms / 1000)  # Thời gian đã trôi qua từ lúc xử lý khung hình
    # print('Elapsed time: {:.2f} seconds'.format(elapsed_time))
    annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)
    cv2.imshow('preprocessed', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    print('result.face_blendshapes[0][9] category_name: {}'.format(result.face_blendshapes[0][9]))
    cv2.moveWindow('preprocessed', 500, 50)
    cv2.waitKey(500)  # Chờ 1ms để hiển thị kết quả
    print_result_callback = functools.partial(print_result, output_image=mp_image, timestamp_ms=frame_timestamp_ms)
    print_result_callback
   

# Options can read on web configuration options of face_landmarker 

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path), # define path of model
    # output_face_blendshapes=True,
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result) # result_callback can only be used in mode LIVE STREAM


with FaceLandmarker.create_from_options(options) as landmarker:
  # The landmarker is initialized. Use it here.
    imcap = cv2.VideoCapture(0)
    imcap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    imcap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        success, frame = imcap.read()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        frame_timestamp_ms = int(imcap.get(cv2.CAP_PROP_POS_MSEC))
        landmarker.detect_async(mp_image, frame_timestamp_ms)
        # cv2.imshow('my capture', frame)
         # Hiển thị kết quả theo thời gian thực
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.breakcap.release()
    cv2.destroyAllWindows('my capture')