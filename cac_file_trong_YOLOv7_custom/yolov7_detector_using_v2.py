import cv2
import torch
import numpy as np
from numpy import random
from models.experimental import attempt_load
from utils.general import check_img_size, set_logging, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Các hàm khác được giữ nguyên
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh.item() - 0.1)), int(round(dh.item() + 0.1))
    # top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw.item() - 0.1)), int(round(dw.item() + 0.1))
    # left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def initialize_model(weights, img_size=640, device='cpu'):
    set_logging()
    device = select_device(device)
    half = device.type != 'cpu'
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    img_size = check_img_size(img_size, s=stride)
    if half:
        model.half()

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(model.parameters())))
    return model, device, half


def detect_objects(model, device, img, half,img_size=640, conf_thres=0.1, iou_thres=0.45):
    # img0 = cv2.imread(img_path)
    img0 = img
    img = letterbox(img0, img_size, stride=model.stride.max())[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3xHxW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0


    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    x_min, y_min, width, height, confidence_formatted = 0, 0, 0, 0, 0.00  # Khởi tạo giá trị mặc định
    with torch.no_grad():
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=[0])
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    x_min, y_min, x_max, y_max = map(int, xyxy)
                    width = x_max - x_min
                    height = y_max - y_min
                    # print('---------------------------------------------------------------------')
                    # print('x, y, w, h:', x_min, y_min, width, height)
                    # label = f'{model.names[int(cls)]} {conf:.2f}'
                    # plot_one_box(xyxy, img0, label='', color='r', line_thickness=3)
    if len(det):
        confidence = conf.item() * 100
    else: confidence = 0.00
    # confidence = conf.item() * 100
    confidence_formatted = "{:.2f}".format(confidence)
    return x_min, y_min, width, height, confidence_formatted
    # cv2.imshow('Predict YOLOv7', img0)
    # cv2.waitKey(0)


# Các đoạn code khởi tạo và sử dụng model YOLOv7 trong file gốc được chuyển thành các hàm tương ứng:
def initialize_and_detect(weights_path, img_path, img_size=640, conf_thres=0.1, iou_thres=0.45, device='cpu'):
    model, device, half = initialize_model(weights_path, img_size, device)
    x, y, w, h = detect_objects(model, device, img_path, half, img_size, conf_thres, iou_thres)
    return x, y, w, h
