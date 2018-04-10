# -*- coding: utf-8 -*-
# file: tools.py
# author: Wang Kang
# time: 12/26/2017 23:16 PM
# ----------------------------------------------------------------
import cv2


def image_show(barname, image):
    """
    :param barname: 图片显示的名称
    :param image: 输入
    :return:
    """
    cv2.namedWindow(barname, cv2.WINDOW_NORMAL)
    cv2.imshow(barname, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
