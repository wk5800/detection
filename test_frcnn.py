#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/3/31/0009 16:32
# @Author  : Wangkang
# @File    : test_frcnn.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy

from __future__ import division
import os, shutil, cv2, sys, pickle, time, requests
import numpy as np
from urllib import request
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from PIL import Image, ImageDraw, ImageFont
import win_unicode_console


win_unicode_console.enable()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parent_dir = 'D:/Image/ELproject/'
my_el = os.path.join(parent_dir, 'my_el/')
clahe_dir = os.path.join(parent_dir, 'clahe_el/')
#predict_el_dir = os.path.join(parent_dir, 'result_el/')
predict_el_dir = os.path.join('//172.16.10.231/Image/ELproject/', 'result_el/')
processed_el_dir = os.path.join(parent_dir, 'processed_el/')

model_dir = 'C:/FRCNN/'
config_output_filename = os.path.join(model_dir, 'model/config.pickle')
model_path = os.path.join(model_dir, 'model/model.hdf5')

font = ImageFont.truetype(os.path.join(model_dir, 'simheittf/simhei.ttf'), 16)  # 字体*.ttc的存放路径
fillColor = (255, 255, 255)  # 标签字体颜色

req_url = 'http://portal.aikosolar.com/aiko-kpi/rest/GetImgservice'  # api

sys.setrecursionlimit(40000)
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# 在测试时关闭任何数据扩充方法
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.num_rois = 10


def format_img_size(img, C):
    """ 根据配置文件格式化图片尺寸 """
    img_min_side = float(C.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, C):
    """ 根据配置文件格式化图片通道"""
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channel_mean[0]
    img[:, :, 1] -= C.img_channel_mean[1]
    img[:, :, 2] -= C.img_channel_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ 对按配置文件配置的预测模型 格式化图像相关信息 """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def get_real_coordinates(ratio, x1, y1, x2, y2):
    """将边界框的坐标转换为初始大小"""
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


def tojava(request_url, json_data):
    """调用api"""
    try:
        req_url = request_url
        r = requests.post(url=req_url, json=json_data)
        print('数据是否发送： ', r.text)
    except request.HTTPError:
        #print("there is an error")
        pass  # 跳过错误，不进行处理，直接继续执行


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
# print(class_mapping)  # 分类种类
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 15) for v in class_mapping}

if C.network == 'resnet50':
    num_features = 1024
elif C.network == 'vgg':
    num_features = 512

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (num_features, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, num_features)

img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

shared_layers = nn.nn_base(img_input, trainable=True)  # 定义基本网络(resnet，VGG等)

num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)  # 定义RPN，它是在基础层上构建的
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

# print('Loading weights from {}'.format(model_path)) 加载模型

model_rpn.load_weights(model_path, by_name=True)
model_classifier.load_weights(model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []
classes = {}
bbox_threshold = 0.7  # 概率大于0.7的显示出来

print('计算已启动...')

while True:
    for i in os.listdir(my_el):  # 对每张图片
        num = os.path.join(my_el, i)
        try:
            time.sleep(0.5)  # 根据离线EL每拍一张图片的时间确定
            original_img = cv2.imread(num)
            gray_img = cv2.imread(num, 0)
            ''''
            # EL旋转
            #img = cv2.imread('flower.jpg')
            rows,cols = gray_img.shape[:2]
            #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
            M = cv2.getRotationMatrix2D((cols/2,rows/2),90,1)
            #第三个参数：变换后的图像大小
            gray_img = cv2.warpAffine(gray_img,M,(rows,cols))
            '''
        except AttributeError:
            print('离线EL拍摄节拍与计算节拍不一致')
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25, 25))
        cl1 = clahe.apply(gray_img)
        file_name = os.path.splitext(i)[0]

        clahe_el_path = clahe_dir + file_name + '.jpg'

        cv2.imwrite(clahe_el_path, cl1)  # 把bmp\jpg格式经过处理后都转化成jpg格式
        time.sleep(1)  # 等待图片clahe处理完毕

        clahe_image = cv2.imread(clahe_el_path)
        X, ratio = format_img(clahe_image, C)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # 从RPN获取特征映射和输出
        [Y1, Y2, F] = model_rpn.predict(X)
        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # 将空间金字塔池应用到 建议的区域
        bboxes = {}
        probs = {}

        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

            for ii in range(P_cls.shape[1]):

                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    probs[cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                try:
                    (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                    tx /= C.classifier_regr_std[0]
                    ty /= C.classifier_regr_std[1]
                    tw /= C.classifier_regr_std[2]
                    th /= C.classifier_regr_std[3]
                    x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
                except:
                    pass
                bboxes[cls_name].append(
                    [C.rpn_stride * x, C.rpn_stride * y, C.rpn_stride * (x + w), C.rpn_stride * (y + h)])
                probs[cls_name].append(np.max(P_cls[0, ii, :]))

        all_dets = []  # 分析列表
        el_dets = []  # 故障列表
        all_dets.append(('imgId', file_name))
        all_dets.append(('elId', '01'))

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]
                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)
                textLabel = '{}:{}'.format(key, '%.2f' % new_probs[jk])  # 文字内容 保留小数点后两位
                el_dets.append((key, '%.2f' % new_probs[jk]))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1,
                                                     1)  # getTextSize()：获取待绘制文本框的大小
                textOrg = (real_x1, real_y1 - 20)  # 文字输出位置

                # 在original_img图片上画出检测对象的轮廓
                cv2.rectangle(original_img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              2)
                img_PIL = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

                # 需要先把输出的中文字符转换成Unicode编码形式
                if not isinstance(textLabel, str):
                    output = textLabel.decode('utf-8')

                draw = ImageDraw.Draw(img_PIL)
                draw.text(textOrg, textLabel, font=font, fill=fillColor)
                original_img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)  # 转换回OpenCV格式
        all_dets.append(('badType', el_dets))
        Img_src = predict_el_dir + 'predicted_%s' % i  # frcnn后的图片
        cv2.imwrite(Img_src, original_img)
        shutil.move(num, processed_el_dir + i)  # 计算后的原图移动到指定目录
        ticks = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))  # 当前时间戳
        all_dets.append(('imgPath', Img_src))
        all_dets.append(('time', ticks))
        data = dict(all_dets)  # list 转化为dict
        # print('all_dets: ', all_dets)
        print('计算结果: ', data)
        #json_str = json.dumps(data, ensure_ascii=False)  # 防止中文变为unicode
        tojava(request_url=req_url, json_data=data)
        #print(json_str)


