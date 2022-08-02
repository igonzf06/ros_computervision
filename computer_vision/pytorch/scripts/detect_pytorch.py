#!/usr/bin/env python3
import cv2
import numpy as np
from zmq import device
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import torch
import torch.onnx
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.models.detection as detection
import torchvision.transforms as transforms
import sys
import requests


class Detector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        # params
        weights = rospy.get_param("/config")
        model_path = rospy.get_param("/model")
        arch = rospy.get_param("/arch")
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

        # load model
        #self.model = torch.hub.load
        entrypoints = torch.hub.list('pytorch/vision', force_reload=True)
        print(entrypoints)
        self.model.load_state_dict(torch.load(weights))
        self.model.eval()
        self.model.to(self.device)

        self.pub = rospy.Publisher(detection_topic, Image, queue_size=10)

        self.bridge = CvBridge()

        # self.subs = rospy.Subscriber(
        #    input_topic, Image, self.callback)

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            #image_height, image_width, _ = image.shape

            blob = cv2.dnn.blobFromImage(
                image, 1.0 / 255, (self.image_size, self.image_size), (0, 0, 0), swapRB=True, crop=False)

            #resize = transforms.Resize((self.image_size, self.image_size))
            #tensor = transforms.ToTensor()
            # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            # std=[0.229, 0.224, 0.225])
            #frame = [torch.tensor(image)]
            #frame = normalize(tensor(resize(image)))
            #frame = frame.to(self.device)
            output = self.model(image)
            print(output)
            # self.model.setInput(blob)
            #output = self.model.forward()

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
