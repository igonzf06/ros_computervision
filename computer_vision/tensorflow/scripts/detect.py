#!/usr/bin/env python3
import cv2
from cv2 import threshold
import numpy as np
import rospy
import torchvision.models as models
import torch.onnx
import torch.nn as nn
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os


class Detector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True,
                        log_level=rospy.INFO)

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
        rospy.loginfo("The class names of the model were loaded.")

        # get a different color array for each of the classes
        self.COLORS = np.random.uniform(
            0, 255, size=(len(self.class_names), 3))

        # load the DNN model
        self.model = cv2.dnn.readNet(
            model=self.model_path, framework=net)

        rospy.loginfo("The model was loaded.")

        self.pub = rospy.Publisher(detection_topic, Image, queue_size=10)

        self.bridge = CvBridge()

        self.subs = rospy.Subscriber(
            input_topic, Image, self.callback)

    def callback(self, data):
        try:

            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_height, image_width, _ = image.shape
            # create blob from image
            blob = cv2.dnn.blobFromImage(
                image, 1.0 / 255, (self.image_size, self.image_size), (0, 0, 0), swapRB=True, crop=False)
            # set the blob to the model
            self.model.setInput(blob)
            # forward pass through the model to carry out the detection
            output = self.model.forward()

            # object detection
            if len(output.shape) > 2:
                for detection in output[0, 0, :, :]:
                    # extract the confidence of the detection
                    confidence = detection[2]

                    # draw bounding boxes only if the detection confidence is above...
                    # ... a certain threshold, else skip
                    if confidence > self.threshold:
                        # get the class id
                        class_id = detection[1]
                        # map the class id to the class
                        class_name = self.class_names[int(class_id)-1]
                        color = self.COLORS[int(class_id)]
                        # get the bounding box coordinates
                        box_x = detection[3] * image_width
                        box_y = detection[4] * image_height
                        # get the bounding box width and height
                        box_width = detection[5] * image_width
                        box_height = detection[6] * image_height
                        # draw a rectangle around each detected object
                        cv2.rectangle(image, (int(box_x), int(box_y)), (int(
                            box_width), int(box_height)), color, thickness=2)
                        # put the FPS text on top of the frame
                        cv2.putText(image, class_name, (int(box_x), int(
                            box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # classification
            else:
                # order the detections
                detections = np.argsort(output[0])[::-1][:5]
                # get the class id to the class
                class_id = detections[0]
                # get the class name
                class_name = self.class_names[int(class_id)-1]
                text = "Label: {}, {:.2f}%".format(
                    class_name, output[0][class_id] * self.base_predict)
                # put the text in the image
                cv2.putText(image, text, (5, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            try:

                self.pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

            except CvBridgeError as e:
                rospy.logerr(e)

        except CvBridgeError as e:
            rospy.logerr(e)


if __name__ == '__main__':

    Detector()
    rospy.spin()
