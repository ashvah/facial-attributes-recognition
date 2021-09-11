#!/usr/bin/env python
# -*- coding:utf8 -*-

import qrcode
import cv2
from PIL import Image
from pyzbar import pyzbar
import numpy as np


def qr_encode(text):
    qr = qrcode.QRCode(
        version=5,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=8,
        border=4)
    qr.add_data(text)
    qr.make(fit=True)
    img_ = qr.make_image()
    return img_


def decode_qr_code_from_path(code_img_path):
    decoded = pyzbar.decode(Image.open(code_img_path), symbols=[pyzbar.ZBarSymbol.QRCODE])
    return list(map(lambda d: d.data.decode(encoding="utf-8"), decoded))


def decode_qr_code_from_cv(img_):
    decoded = pyzbar.decode(img_, symbols=[pyzbar.ZBarSymbol.QRCODE])
    return list(map(lambda d: d.data.decode(encoding="utf-8"), decoded))


if __name__ == "__main__":
    cv2.namedWindow('window_frame')
    video_capture = cv2.VideoCapture(0)
    print(video_capture)
    while True:
        bgr_image = video_capture.read()[1]
        bgr_image = cv2.flip(bgr_image, 1)

        mask = np.zeros(bgr_image.shape, dtype=np.uint8)
        bgr_image = cv2.illuminationChange(bgr_image, mask)

        res = decode_qr_code_from_cv(bgr_image)
        if not res:
            print("No qrcode!")
        else:
            [print(r) for r in res]

        cv2.imshow('window_frame', bgr_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # res = decode_qr_code_from_path("./2.png")
    # if not res:
    #     print("No qrcode!")
    # else:
    #     [print(r) for r in res]
    cv2.destroyAllWindows()
