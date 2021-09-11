#!/usr/bin/env python
# -*- coding:utf8 -*-

import cv2
from facenet_pytorch import MTCNN
from PIL import Image
import torch
import numpy as np
import torchvision
import json
from classification.utils import predict, load_model, get_transform


def put_text(img_array, glass, hair, point, color_, font=cv2.FONT_HERSHEY_SIMPLEX):
    # if hair and glass:
    #     cv2.putText(img_array, "Glasses Long", point, font, 0.5, color_, 1)
    # elif glass and not hair:
    #     cv2.putText(img_array, "Glasses Short", point, font, 0.5, color_, 1)
    # elif not glass and hair:
    #     cv2.putText(img_array, "No Glasses Long", point, font, 0.5, color_, 1)
    # else:
    #     cv2.putText(img_array, "No Glasses Short", point, font, 0.5, color_, 1)
    cv2.putText(img_array, "Glass: {:.4} Long: {:.4}".format(glass, hair), point, font, 0.5, color_, 1)


def draw_bounding_box(face_coordinates_, image_array, color_):
    x1_, y1_, x2_, y2_ = face_coordinates_
    cv2.rectangle(image_array, (x1_, y1_), (x2_, y2_), color_, 2)


if __name__ == '__main__':
    cv2.namedWindow('window_frame')
    # cv2.namedWindow('equalized_image')
    color = (0, 255, 0)
    video_capture = cv2.VideoCapture(0)
    glass_classifier = load_model("classification/weights/Res", name='Res', num_of_classes=1)
    hair_classifier = load_model("classification/weights/Res_hair", name='Res3', num_of_classes=1)
    # torch.save(glass_classifier.state_dict(), "./weight0.9999_99999.pt", _use_new_zipfile_serialization=False)
    # torch.save(hair_classifier.state_dict(), "./hair.pt", _use_new_zipfile_serialization=False)
    mtcnn = MTCNN()
    trans = get_transform('lfw_5590')
    while True:
        bgr_image = video_capture.read()[1]
        # bgr_image = cv2.imread("./3.jpg")
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        # gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HLS)
        # gray_image[:, :, 1] = cv2.equalizeHist(gray_image[:, :, 1])
        # rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_HLS2RGB)

        img = Image.fromarray(rgb_image)
        origin_size = img.size

        faces, _ = mtcnn.detect(img)

        final_result = {}
        if faces is not None:
            for index, face_coordinates in enumerate(faces):
                x1, y1, x2, y2 = face_coordinates
                width = x2 - x1
                height = y2 - y1

                x11 = int(max(x1 - width * 0.30, 0))
                y11 = int(max(y1 - height * 0.35, 0))
                x22 = int(min(x2 + width * 0.30, origin_size[0]))
                y22 = int(min(y2 + height * 0.25, origin_size[1]))
                img_ = img.crop((x11, y11, x22, y22))

                # data1 = trans[0](img_).reshape(1, 3, 224, 224)

                # img_cv = cv2.cvtColor(np.asarray(img_), cv2.COLOR_RGB2HLS)
                # img_cv[:, :, 1] = cv2.equalizeHist(img_cv[:, :, 1])
                # img_cv = cv2.cvtColor(img_cv, cv2.COLOR_HLS2RGB)
                # cv2.imshow('equalized_image', cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR))
                # img_ = Image.fromarray(img_cv)

                data = trans[0](img_).reshape(1, 3, 224, 224)
                # data2 = trans[0](img_).reshape(1, 3, 224, 224)
                # data3 = trans[0](img_).reshape(1, 3, 224, 224)
                # data = torch.cat(img_, dim=0)

                draw_bounding_box((x1, y1, x2, y2), rgb_image, color)
                # img_ = img.crop((x11, y11, x22, y22))

                # data = trans[0](img_).reshape(1, 3, 224, 224)
                # data2 = trans[0](img_).reshape(1, 3, 224, 224)
                # data3 = trans[0](img_).reshape(1, 3, 224, 224)
                # data = torch.cat([data1, data2, data3], dim=0)
                glass = predict(glass_classifier, data, mean=True).squeeze()
                hair = predict(hair_classifier, data, mean=True).squeeze()

                # glass1 = predict(glass_classifier, data1, mean=True).squeeze()
                # hair1 = predict(hair_classifier, data1, mean=True).squeeze()

                # put_text(rgb_image, glass1, hair1, (int(x1 + 10), int(y1 + 20)), color_=(0, 0, 255))

                put_text(rgb_image, glass, hair, (int(x1 + 10), int(y1 + 10)), color_=color)
                final_result["face_" + str(index)] = {"glass": glass.item(), "hair": hair.item()}

        json_encode = json.dumps(final_result)
        print(json_encode)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('window_frame', bgr_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
