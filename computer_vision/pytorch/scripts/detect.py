#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import torch
import torch.onnx
import torch.nn as nn
import torchvision
import torchvision.models as models
import sys
import requests


class Detector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        # params
        net = rospy.get_param("/net")
        self.model_path = rospy.get_param("/model")
        self.arch = rospy.get_param("/arch")
        self.device = rospy.get_param("/device")
        self.threshold = rospy.get_param("/threshold")
        self.image_size = rospy.get_param("/image_size")
        classes = rospy.get_param("/classes")
        input_topic = rospy.get_param("/input_topic")
        detection_topic = rospy.get_param("/detection_topic")

        # load the class names
        with open(classes, 'r') as f:
            self.class_names = f.read().split('\n')

        # get a different color array for each of the classes
        self.COLORS = np.random.uniform(
            0, 255, size=(len(self.class_names), 3))

        # Conver pytorch model to onnx model to opencv
        model_path = "/home/irene/tfm_ros_ws/src/room_detection/models/frame_based/resnet50/resnet50_action.onnx"
        model = models.__dict__['resnet50'](pretrained=False)
        print(model)
        ''' model.fc = nn.Linear(2048, len(self.class_names))
        try:
            model.load_state_dict(torch.load(
                '/home/irene/tfm_ros_ws/src/room_detection/models/frame_based/resnet50/best_classifier.pth', map_location=torch.device('cpu')))
            model.eval()
            input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model, input, model_path, verbose=True)
        except:
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(
                '/home/irene/tfm_ros_ws/src/room_detection/models/frame_based/resnet50/best_classifier.pth', map_location=torch.device('cpu')))

            model.eval()
            input = torch.randn(1, 3, 224, 224)
            torch.onnx.export(model.module, input, model_path, verbose=True)
 '''
        self.model = cv2.dnn.readNetFromONNX(model_path)

        self.pub = rospy.Publisher(detection_topic, Image, queue_size=10)

        self.bridge = CvBridge()

        self.subs = rospy.Subscriber(
            input_topic, Image, self.callback)

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_height, image_width, _ = image.shape

            blob = cv2.dnn.blobFromImage(
                image, 1.0 / 255, (self.image_size, self.image_size), (0, 0, 0), swapRB=True, crop=False)

            self.model.setInput(blob)
            output = self.model.forward()

            #best_prediction = np.array(predictions)[0].argmax()
            detections = np.argsort(output[0])[::-1][:5]

            #class_name = self.class_names[int(best_prediction)-1]
            class_id = detections[0]
            # get the class name
            class_name = self.class_names[int(class_id)-1]
            text = "Label: {}, {:.2f}%".format(
                class_name, output[0][class_id] * 10)
            # put the text in the image
            cv2.putText(image, text, (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            try:
                self.pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            except CvBridgeError as e:
                print(e)

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':

    Detector()
    rospy.spin()
