import argparse

import cv2
import torch
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def draw_bbox(image: np.array,
              bbox: np.array,
              label: str | None = None) -> np.array:
    """
    Draw one bounding box on image

    :param image: np.array
    :param bbox: np.array or list or tuple. List like (x1, y1, x2, y2)
    :param label: str
    :return: np.array
    """
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)
    image = cv2.rectangle(image, bbox[0:2], bbox[2:4],
                          color=RED,
                          thickness=3)
    if label is not None:
        (w, h), _ = cv2.getTextSize(label,
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1.5,
                                    thickness=3)
        x, y = bbox[0:2]
        x1, y1 = x, y - h
        x2, y2 = x + w, y
        image = cv2.rectangle(image, (x1, y1), (x2, y2),
                              color=RED,
                              thickness=-1)

        image = cv2.putText(image, label,
                            org=bbox[0:2],
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1.5,
                            color=WHITE,
                            thickness=3,
                            lineType=cv2.LINE_AA)
    return image


def draw_bbox_on_objects(image: np.array,
                         table: pd.DataFrame,
                         favorite: list | None = None) -> np.array:
    """
    Draw several bounding box on image.

    In input table must be coordinates left top and right bottom corners bounding box,
    also class name and confidence.
    Table fields must be like (xmin, ymin, xmax, ymax, name, confidence)

    :param image: np.array
    :param table: pd.DataFrame
    :param favorite: list, class name for drawing
    :return: np.array
    """
    image = image.copy()
    for idx in range(table.shape[0]):
        line = table.iloc[idx]

        lab = line['name']
        if favorite is not None and lab not in favorite:
            continue
        conf = line['confidence']
        bbox = (int(line['xmin']), int(line['ymin']), int(line['xmax']), int(line['ymax']))
        label = "{}: {:.2f}".format(lab, conf)
        image = draw_bbox(image=image, bbox=bbox, label=label)
    return image


def load_model(path_to_weights: str):
    """
    Load YOLOv5 from ultralytics with parameter 'model' equal 'custom'
    and load model parameters by 'path_to_weights' path.

    :param path_to_weights: str, path to model
    :return: torch.nn.Module
    """
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=path_to_weights)
    model.eval()

    return model


def detect(path_to_video: str,
           path_to_save: str,
           path_to_weights: str):
    """
    Detect human on receive video

    :param path_to_video: str, path to video on device
    :param path_to_save: str, path for save video with drawed bounding boxes on device
    :param path_to_weights: str, path to model saved in *.pt file
    :return: None
    """

    model = load_model(path_to_weights)

    cap = cv2.VideoCapture(path_to_video)

    param = {
        'frameSize': (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
        'fps': 25
    }
    video = cv2.VideoWriter(path_to_save, **param)

    have_frame, frame = cap.read()
    while have_frame:
        table = model(frame).pandas().xyxy[0]
        frame = draw_bbox_on_objects(frame, table, favorite=['person'])
        video.write(frame)
        have_frame, frame = cap.read()

    cap.release()
    video.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_video', type=str)
    parser.add_argument('path_to_save', type=str)
    parser.add_argument('-w', nargs='?', default='models/yolov5s6.pt')

    var = parser.parse_args()

    path_to_video = var.path_to_video
    path_to_save = var.path_to_save
    path_to_model = var.w

    detect(path_to_video, path_to_save, path_to_model)
