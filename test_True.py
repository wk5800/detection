from __future__ import division
import os, shutil, json
import cv2
import numpy as np
import sys
import pickle
import time
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
from PIL import Image, ImageDraw, ImageFont

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
my_el = 'C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/image/my_el'
clahe_dir = 'C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/image/clahe_el/'
predict_el_dir = 'C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/image/predict_el/'
processed_el_dir = 'C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/image/processed_el/'
sys.setrecursionlimit(40000)
config_output_filename = "C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/dataset/config.pickle"
with open(config_output_filename, 'rb') as f_in:
    C = pickle.load(f_in)

if C.network == 'resnet50':
    import keras_frcnn.resnet as nn
elif C.network == 'vgg':
    import keras_frcnn.vgg as nn

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False
C.num_rois = 10
C.model_path = "C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/dataset/model_frcnn.hdf5"

def format_img_size(img, C):
    """ formats the image size based on config """
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
    """ formats the image channels based on config """
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
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


# Method to transform the coordinates of the bounding box to its original size
def get_real_coordinates(ratio, x1, y1, x2, y2):
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return (real_x1, real_y1, real_x2, real_y2)


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}
print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}

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

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

print('Loading weights from {}'.format(C.model_path))
model_rpn.load_weights(C.model_path, by_name=True)  # 加载模型
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = 0.5  # 概率大于0.7的显示出来

while True:
    for i in os.listdir(my_el):  # 对每张图片
        num = os.path.join(my_el, i)

        try:
            time.sleep(0.2)  # 根据离线EL每拍一张图片的时间确定
            '''
            # EL旋转
            original_img = cv2.imread(num, 0)
            rows, cols = original_img.shape[:2]
            #第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
            M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
            #第三个参数：变换后的图像大小
            gray_img = cv2.warpAffine(original_img, M, (rows, cols))
            '''

            original_img = cv2.imread(num)
            gray_img = cv2.imread(num, 0)

        except AttributeError:
            print('离线EL拍摄节拍与计算节拍不一致')

        file_name = os.path.splitext(i)[0]

        # 对输入图片预处理
        '''
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25, 25))
        cl1 = clahe.apply(gray_img)

        clahe_el_path = clahe_dir + i

        cv2.imwrite(clahe_el_path, cl1)  # 把bmp\jpg格式经过处理后都转化成jpg格式
        time.sleep(0.2)  # 等待图片clahe处理完毕

        '''
        clahe_el_path = num  # 输入图片不做clahe预处理


        clahe_image = cv2.imread(clahe_el_path)

        X, ratio = format_img(clahe_image, C)
        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))

        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] -= R[:, 0]
        R[:, 3] -= R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
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

        all_dets = []

        all_dets.append(('ID', file_name))

        for key in bboxes:
            bbox = np.array(bboxes[key])

            new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.2)  # overlap_thresh为非极大值抑制过程中过滤掉重合度为overlap_thresh的框
            for jk in range(new_boxes.shape[0]):
                (x1, y1, x2, y2) = new_boxes[jk, :]

                (real_x1, real_y1, real_x2, real_y2) = get_real_coordinates(ratio, x1, y1, x2, y2)

                textLabel = '{}:{}'.format(key, '%.2f' % new_probs[jk])  # 文字内容 保留小数点后两位

                all_dets.append(('EL_type'+str(jk), (key, '%.2f' % new_probs[jk])))

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1,
                                                     1)  # getTextSize()：获取待绘制文本框的大小
                textOrg = (real_x1, real_y1 - 20)  # 文字输出位置

                # 在original_img图片上画出检测对象的轮廓
                cv2.rectangle(original_img, (real_x1, real_y1), (real_x2, real_y2),
                              (int(class_to_color[key][0]), int(class_to_color[key][1]), int(class_to_color[key][2])),
                              2)

                img_PIL = Image.fromarray(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))

                # 字体 字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
                font = ImageFont.truetype(
                    'C:/pythonProject/CosmicadDetection-Keras-Tensorflow-FasterRCNN-master/simheittf/simhei.ttf', 16)

                fillColor = (255, 255, 255)  # 标签字体颜色

                # 需要先把输出的中文字符转换成Unicode编码形式
                if not isinstance(textLabel, str):
                    output = textLabel.decode('utf-8')

                draw = ImageDraw.Draw(img_PIL)
                draw.text(textOrg, textLabel, font=font, fill=fillColor)

                # 转换回OpenCV格式
                original_img = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)

        Img_src = predict_el_dir + 'predicted_%s' % i
        cv2.imwrite(Img_src, original_img)
        shutil.move(num, processed_el_dir + i)
        ticks = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))  # 当前时间戳
        all_dets.append(('Img_src', Img_src))
        all_dets.append(('Process_time', ticks))
        data = dict(all_dets)  # list 转化为dict
        json_str = json.dumps(data, ensure_ascii=False)  # 防止中文变为unicode
        print('Result: ', json_str)
