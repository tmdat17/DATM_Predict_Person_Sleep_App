import cv2
import torch
import numpy as np
from numpy import random
from YOLOv7.models.experimental import attempt_load
from YOLOv7.utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from YOLOv7.utils.plots import plot_one_box
from YOLOv7.utils.torch_utils import select_device, time_synchronized

# Sử dụng class YOLOv7Detector
weights_path = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt"
# source_image_path = "E:\CTU\LUAN_VAN_2023\sample_img_pose\pose_sit_wi_1_resize.jpg"


class YOLOv7Detector:
    def __init__(self, weights = "E:\CTU\LUAN_VAN_2023\YOLOv7\pretrain\yolov7.pt", img_size=640, conf_thres=0.1, iou_thres=0.45, device='cpu', classes=None):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.classes = classes
        
        self.model = None
        self.names = None
        self.colors = None

    def initialize_model(self):
        set_logging()
        device = select_device(self.device)
        self.model = attempt_load(self.weights, map_location=device)
        stride = int(self.model.stride.max())
        self.img_size = check_img_size(self.img_size, s=stride)
        if device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.img_size, self.img_size).to(device).type_as(next(self.model.parameters())))

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

    def detect_objects(self, source_image_path):
        img0 = cv2.imread(source_image_path)
        # img = letterbox(img0, self.img_size, stride=self.model.stride.max())[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if self.classes:
            classes = []
            for class_name in self.classes:
                classes.append(self.classes.index(class_name))

        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=classes, agnostic=False)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    label = f"{n} {self.names[int(c)]}{'s' * (n > 1)}"
                    for *xyxy, conf, cls in reversed(det):
                        x_min, y_min, x_max, y_max = map(int, xyxy)  # Lấy giá trị tọa độ x_min, y_min, x_max, y_max
                        width = x_max - x_min  # Tính toán chiều rộng
                        height = y_max - y_min  # Tính toán chiều dài
                        print('---------------------------------------------------------------------')
                        print('x, y, w, h:', x_min, y_min, width, height)
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=3)

        cv2.imshow('predict YOLOv7', img0)
        cv2.waitKey(0)
        
# Đoạn này sẽ được gọi bên file khác với 3 dòng bên dưới
# detector = YOLOv7Detector(weights_path, classes=['person'])
# detector.initialize_model()
# detector.detect_objects(source_image_path)