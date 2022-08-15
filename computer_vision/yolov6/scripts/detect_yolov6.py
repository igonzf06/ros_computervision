#!/usr/bin/env python3
from charset_normalizer import detect
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch
import torch.onnx
from tqdm import tqdm
from yolov6.utils.events import LOGGER, load_yaml
import PIL

from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression


class Detector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.weights = "/home/irene/tfm_ros_ws/src/room_detection/models/yolov6/yolov6s.pt"
        self.device = "cpu"
        self.yaml = "data/coco.yaml"
        self.half = False
        self.class_names = load_yaml(self.yaml)['names']

        # get a different color array for each of the classes
        self.COLORS = np.random.uniform(
            0, 255, size=(len(self.class_names), 3))
        self.model = DetectBackend(self.weights, device=self.device)

        self.pub = rospy.Publisher("image_detection", Image, queue_size=10)

        self.bridge = CvBridge()

        self.subs = rospy.Subscriber(
            "/usb_cam/image_raw", Image, self.callback)

    def callback(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_height, image_width, _ = img.shape

            conf_thres = 0.25
            iou_thres = 0.45
            classes = None
            agnostic_nms = False
            max_det = 1000
            hide_labels = False
            hide_conf = False

            ''' Model Inference and results visualization '''

            image = letterbox(img, image_height, stride=self.model.stride)[0]

            # Convert
            image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            image = torch.from_numpy(np.ascontiguousarray(image))
            image = image.half() if self.half else image.float()  # uint8 to fp16/32
            image /= 255  # 0 - 255 to 0.0 - 1.0

            if len(image.shape) == 3:
                image = image[None]

            # predictions
            pred_results = self.model(image)
            det = non_max_suppression(
                pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)[0]

            if len(det) > 0:
                det[:, :4] = self.rescale(
                    image.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in reversed(det):

                    class_num = int(cls)
                    label = None if hide_labels else (
                        self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')
                    box = xyxy
                    color = self.COLORS[int(class_num)]

                    x = int(box[0])
                    y = int(box[1])
                    w = int(box[2])
                    h = int(box[3])

                    cv2.rectangle(img, (x, y),
                                  (w, h), color, thickness=2)
                    cv2.putText(img, label, (int(x), int(y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            try:
                self.pub.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))
            except CvBridgeError as e:
                print(e)

        except CvBridgeError as e:
            print(e)

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0],
                    ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / \
            2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes


if __name__ == '__main__':

    Detector()
    rospy.spin()
