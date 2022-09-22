#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image
from caffe.msg import Prediction, Predictions, BoundingBox
from cv_bridge import CvBridge, CvBridgeError
import os


class Visualizer:
    def __init__(self):
        rospy.init_node('object_visualizer', anonymous=True)

        # params
        classes = rospy.get_param("/classes")
        detection_topic = rospy.get_param("/detection_topic")
        self.image_size = rospy.get_param("/image_size")
        self.threshold = rospy.get_param("/threshold")
        input_topic = rospy.get_param("/input_topic")

        # load the COCO class names
        with open(classes, 'r') as f:
            self.class_names = f.read().split('\n')

        # get a different color array for each of the classes
        self.COLORS = np.random.uniform(
            0, 255, size=(len(self.class_names), 3))

        self._pub = rospy.Publisher(detection_topic, Image, queue_size=100)

        self.bridge = CvBridge()

        self._subs = message_filters.Subscriber(
            input_topic, Image)
        self._subsP = message_filters.Subscriber(
            "/predictions", Predictions)

        self._ts = message_filters.ApproximateTimeSynchronizer(
            [self._subs, self._subsP], 10, 0.1)
        self._ts.registerCallback(self.callback)

    def callback(self, image_data, predictions):
        try:

            image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")
            image_height, image_width, _ = image.shape

            for pred in predictions.Predictions:
                class_name = pred.label
                color = self.COLORS[0]
                print(class_name)
                # get the bounding box coordinates
                boundingBox = pred.boundingBox
                # draw a rectangle around each detected object
                cv2.rectangle(image, (int(boundingBox.x), int(boundingBox.y)), (
                    int(boundingBox.width), int(boundingBox.height)), color, thickness=2)
                # put the FPS text on top of the frame
                cv2.putText(image, class_name, (int(boundingBox.x),
                            int(boundingBox.y)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            self._pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':

    Visualizer()
    rospy.spin()
