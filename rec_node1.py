#!/usr/bin/env python
# -*- coding:utf8 -*-


import rospy
from std_srvs.srv import Trigger, TriggerResponse
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import cv2
from PIL import Image as Img
from facenet_pytorch import MTCNN
import json
from classification.utils import predict, load_model, get_transform
import torch
import numpy as np


class FacialAttr():
    def __init__(self):
        rospy.init_node('facial_attr')
        self.path = rospy.get_param("~path", "/home/ucar/ucar_ws/src/face_attr_detection")
        self.up_margin = rospy.get_param("~up", 0.35)
        self.down_margin = rospy.get_param("~down", 0.20)
        self.left_margin = rospy.get_param("~left", 0.35)
        self.right_margin = rospy.get_param("~right", 0.25)
        self.detector = MTCNN(device=torch.device('cuda'))
        self.glass_classifier = load_model(self.path + "/src/classification/weights/Res_glass", name='Res3',
                                           num_of_classes=1)
        self.hair_classifier = load_model(self.path + "/src/classification/weights/Res", name='Res',
                                          num_of_classes=1)
        self.trans = get_transform('lfw_5590')[0]

        self.bridge = CvBridge()
        # self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.img_callback)
        # self.call = False
        # self.cv_image = []

        self.server = rospy.Service('/facial_attr_prediction', Trigger, self.predict_callback)

    # def img_callback(self, msg):
    #     if not self.call:
    #         self.cv_image = []
    #         return
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #         rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    #         img = Img.fromarray(rgb_image)
    #     except CvBridgeError as e:
    #         rospy.logerr("Failed to transforn image from ros to cv")
    #         print(e)
    #     self.cv_image.append(img)

    def predict_once(self):
        img = rospy.wait_for_message("/usb_cam/image_raw", Image)
        try:
            cv_image = self.bridge.imgmsg_to_cv2(img, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("Failed to transforn image from ros to cv")
            print(e)
            return None, False
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img = Img.fromarray(rgb_image)
        # while not self.cv_image:
        #     rospy.sleep(0.1)
        origin_size = img.size
        # self.call=False

        print("before detecting faces")
        faces, _ = self.detector.detect(img)
        print("after detecting")

        final_result = {}
        if faces is not None:
            for index, face_coordinates in enumerate(faces):
                x1, y1, x2, y2 = face_coordinates
                width = x2 - x1
                height = y2 - y1

                img_ = []
                for i in range(5):
                    x11 = int(max(x1 - width * (self.left_margin - i * 0.05), 0))
                    y11 = int(max(y1 - height * (self.up_margin - i * 0.05), 0))
                    x22 = int(min(x2 + width * (self.right_margin - i * 0.05), origin_size[0]))
                    y22 = int(min(y2 + height * (self.down_margin - i * 0.05), origin_size[1]))
                    img_.append(self.trans[0](img.crop((x11, y11, x22, y22))).reshape(1, 3, 224, 224))

                # data = self.trans(img_).reshape(1, 3, 224, 224)
                # data2 = trans[0](img_).reshape(1, 3, 224, 224)
                # data3 = trans[0](img_).reshape(1, 3, 224, 224)
                data = torch.cat(img_, dim=0)
                glass = predict(self.glass_classifier, data, mean=True).squeeze()
                hair = predict(self.hair_classifier, data, mean=True).squeeze()
                final_result["face_" + str(index)] = {"glass": glass.item(), "hair": hair.item()}
        else:
            return None, True
        json_encode = json.dumps(final_result)
        print(json_encode)

        # 解码方式
        # json_decode = json.loads(json_encode)
        # print(json_decode["face_0"]["glass"])

        return json_encode, True

    def predict_callback(self, req):
        res, success = self.predict_once()
        if not success:
            return TriggerResponse(0, "Failed to convert the format of image!")
        if res is None:
            return TriggerResponse(0, "Failed to detect anything!")
        return TriggerResponse(1, res)


if __name__ == "__main__":
    f = FacialAttr()
    f.predict_once()
    f.predict_once()
    f.predict_once()
    try:
        while not rospy.is_shutdown():
            rospy.sleep(0.1)

    except rospy.ROSInterruptException:
        pass
