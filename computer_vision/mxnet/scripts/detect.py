#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import os
import mxnet as mx
from collections import namedtuple
Batch = namedtuple('Batch', ['data'])


class Detector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        # params
        self.model_prefix = rospy.get_param("/prefix")
        self.model_epoch = rospy.get_param("/epoch")
        self.device = rospy.get_param("/device")
        classes = rospy.get_param("/classes")
        input_topic = rospy.get_param("/input_topic")
        detection_topic = rospy.get_param("/detection_topic")
        self.image_size = rospy.get_param("/image_size")

        # load the model class names
        with open(classes, 'r') as f:
            self.class_names = f.read().split('\n')

        # get a different color array for each of the classes
        self.COLORS = np.random.uniform(
            0, 255, size=(len(self.class_names), 3))

        # load the MxNet model
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            self.model_prefix, self.model_epoch)

        self.mod = mx.mod.Module(
            symbol=sym, context=mx.cpu(), label_names=None)

        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.image_size, self.image_size))],
                      label_shapes=self.mod._label_shapes)

        self.mod.set_params(arg_params, aux_params, allow_missing=True)

        self.pub = rospy.Publisher(detection_topic, Image, queue_size=10)

        self.bridge = CvBridge()

        self.subs = rospy.Subscriber(
            input_topic, Image, self.callback)

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            image_height, image_width, _ = image.shape

            img = cv2.resize(image, (224, 224))
            img = np.swapaxes(img, 0, 2)
            img = np.swapaxes(img, 1, 2)
            img = img[np.newaxis, :]
            # forward pass through the model to carry out the detection
            self.mod.forward(Batch([mx.nd.array(img)]))
            prob = self.mod.get_outputs()[0].asnumpy()
            print(self.mod.get_outputs())
            print("shape")
            print(len(prob.shape))
            prob = np.squeeze(prob)
            a = np.argsort(prob)[::-1]
            i = a[0]
            print('probability=%f, class=%s' %
                  (prob[i], self.class_names[i]))
            class_name = self.class_names[i]
            color = self.COLORS[i]
            text = "{}, {:.2f}%".format(
                class_name, prob[i])
            cv2.putText(image, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            try:
                self.pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
            except CvBridgeError as e:
                print(e)

        except CvBridgeError as e:
            print(e)


if __name__ == '__main__':

    Detector()
    rospy.spin()
