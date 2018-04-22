from __future__ import absolute_import
import numpy as np
import cv2
import random
import copy
from . import data_augment
import threading
import itertools

'''
IOU，RPN计算
'''


def union(au, bu, area_intersection):
    """并集"""
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection
    return area_union


def intersection(ai, bi):
    """交集"""
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    """IOU(交并比)"""
    # a and b should be (x1,y1,x2,y2)

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def get_new_img_size(width, height, img_min_side=600):
    """图像resize 最短的边resize为600"""
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


class SampleSelector:
    def __init__(self, class_count):
        # ignore classes that have zero samples
        self.classes = [b for b in class_count.keys() if class_count[b] > 0]
        self.class_cycle = itertools.cycle(self.classes)
        self.curr_class = next(self.class_cycle)

    def skip_sample_for_balanced_class(self, img_data):

        class_in_img = False

        for bbox in img_data['bboxes']:

            cls_name = bbox['class']

            if cls_name == self.curr_class:
                class_in_img = True
                self.curr_class = next(self.class_cycle)
                break

        if class_in_img:
            return False
        else:
            return True


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    """
    计算RPN
    :param C:配置信息
    :param img_data:包含一张图片的路径，bbox坐标和对应的分类（可能一张图片有多组，即表示图片里包含多个对象）
    :param width:原尺寸
    :param height:原尺寸
    :param resized_width:resize之后的尺寸,用于求bbox坐标在resize之后图片上的坐标
    :param resized_height:resize之后的尺寸,用于求bbox坐标在resize之后图片上的坐标
    :param img_length_calc_function:一个方法，基于我们的设置来从图片尺寸计算出经过网络之后特征图的尺寸
    :return:是否包含类,相应的回归梯度
    """
    downscale = float(C.rpn_stride)  # 从图片到特征图的缩放倍数
    anchor_sizes = C.anchor_box_scales  # 初步选区大小的参数  三个
    anchor_ratios = C.anchor_box_ratios  # 初步选区大小的参数  三个
    num_anchors = len(anchor_sizes) * len(anchor_ratios)

    # 根据网络体系结构计算特征图大小。
    (output_width, output_height) = img_length_calc_function(resized_width, resized_height)  # 计算特征图的尺寸

    n_anchratios = len(anchor_ratios)

    # 初始化输出目标
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))

    num_bboxes = len(img_data['bboxes'])  # 每个图片的bboxes数量

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)

    # 把bbox resize以后的图像记做gta
    gta = np.zeros((num_bboxes, 4))  # gta，尺寸为(num_of_bbox,4)
    for bbox_num, bbox in enumerate(img_data['bboxes']):
        """把bbox中的x1,x2,y1,y2分别通过缩放匹配到resize以后的图像,记做gta"""
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # rpn ground truth
    for anchor_size_idx in range(len(anchor_sizes)):
        """计算anchor的长宽"""

        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]

            for ix in range(output_width):
                # x-coordinates of the current anchor box
                """把特征图的每一个点作为一个锚点，通过乘以downscale，映射到图片的实际尺寸"""
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # ignore boxes that go across image boundaries
                if x1_anc < 0 or x2_anc > resized_width:
                    """ 再结合anchor的尺寸，忽略掉超出图片范围的"""
                    continue

                for jy in range(output_height):
                    """把特征图的每一个点作为一个anchor，通过乘以downscale，映射到图片的实际尺寸"""
                    # y-coordinates of the current anchor box
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # ignore boxes that go across image boundaries
                    if y1_anc < 0 or y2_anc > resized_height:
                        """再结合anchor的尺寸，忽略掉超出图片范围的"""
                        continue

                    # bbox_type indicates whether an anchor should be a target
                    bbox_type = 'neg'

                    # this is the best IOU for the (x,y) coord and the current anchor
                    # note that this is different from the best IOU for a GT bbox
                    best_iou_for_loc = 0.0  # ioc最佳交集

                    """
                    现在我们确定了一个预选框组合有确定了中心点那就是唯一确定一个框了，
                    接下来就是来确定这个宽的性质了：是否包含物体、如包含物体其回归梯度是多少?

                    要确定以上两个性质，每一个框都需要遍历图中的所有bboxes
                    """
                    for bbox_num in range(num_bboxes):
                        """遍历大小、比例不一的矩形选框"""

                        # 计算了该anchole预选框与gta（bbox）的交集 IOU
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # calculate the regression targets if they will be needed
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            """如果现在的交并比curr_iou大于该bbox最好的交并比或者大于给定的阈值，则求下列参数，这些参数是后来要用的即回归梯度"""
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0  # 计算gta的中心点坐标
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0  # 计算anchor的中心点坐标
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)  # 根据中心点坐标，计算x的梯度值
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log(
                                (gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))  # 根据resize后bbox坐标，计算x的梯度值
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['bboxes'][bbox_num]['class'] != 'bg':
                            """根据anchor的表现对其进行标注"""

                            # all GT boxes should be mapped to an anchor box, so we keep track of which anchor box was best
                            if curr_iou > best_iou_for_bbox[bbox_num]:
                                """如果交集大于当前这个bbox（gta）的最佳值，则进行一系列更新（更新当前gta的最匹配anchor，最大交集，最优region坐标，最优梯度）"""
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # we set the anchor to positive if the IOU is >0.7 (it does not matter if there was another better box, it just indicates overlap)
                            if curr_iou > C.rpn_max_overlap:
                                """如果交集大于设定的阈值，则bbox_type定义为一个positive的anchor，即存在与之重合度比较高的bbox，同时该bbox的num_anchors加1。"""
                                bbox_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1
                                # we update the regression layer target if this IOU is the best for the current (x,y) and anchor position
                                if curr_iou > best_iou_for_loc:
                                    """如果交集刚好也大于best_iou_for_loc（该anchor下的iou最佳交集），则更新best_iou_for_loc（ioc最佳交集）和最佳梯度。"""
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # if the IOU is >0.3 and <0.7, it is ambiguous and no included in the objective
                            if C.rpn_min_overlap < curr_iou < C.rpn_max_overlap:
                                # 如果刚好处于最大（0.7）和最小阈值（0.3）之间，那不确定它是背景还是对象，将其定义为neutral，即中性。
                                if bbox_type != 'pos':
                                    bbox_type = 'neutral'

                    # 对anchor进行打标
                    if bbox_type == 'neg':
                        """neg,负样本"""
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'neutral':
                        """neutral,中立样本"""
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif bbox_type == 'pos':
                        """pos,正样本"""
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1

                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # 我们确保每个bbox至少有一个正RPN区域

    for idx in range(num_anchors_for_bbox.shape[0]):
        """根据bbox_type对本anchor进行打标，y_is_box_valid和y_rpn_overlap分别定义了这个anchor是否可用和是否包含对象。"""
        if num_anchors_for_bbox[idx] == 0:
            # no box with an IOU greater than zero ...
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], best_anchor_for_bbox[
                    idx, 2] + n_anchratios *
                best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
            best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    # 保证每个bbox至少有一个anchor与之对应。
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)

    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_pos = len(pos_locs[0])

    # 一个问题是，RPN比正区域有更多的neg区域，所以我们关闭了一些neg的区域。
    # regions. We also limit it to 256 regions.

    # 对pos和neg的anchor进行了定位
    num_regions = 256  # 返回num_regions（这里设置为256）个有效的正负样本

    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(len(pos_locs[0])), len(pos_locs[0]) - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    # 因为negtive的anchor肯定远多于postive的，因此在这里设定了regions数量的最大值，并对pos和neg的样本进行了均匀的取样
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)  # 别用于确定anchor是否包含物体(类)，和回归梯度。其只会返回num_regions（这里设置为256）个有效的正负样本


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """

    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))

    return g


def get_anchor_gt(all_img_data, class_count, C, img_length_calc_function, backend, mode='train'):
    # The following line is not useful with Python 3.5, it is kept for the legacy
    # all_img_data = sorted(all_img_data)

    sample_selector = SampleSelector(class_count)

    while True:
        if mode == 'train':
            np.random.shuffle(all_img_data)

        for img_data in all_img_data:
            try:

                if C.balanced_classes and sample_selector.skip_sample_for_balanced_class(img_data):
                    continue

                # read in image, and optionally add augmentation

                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=True)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                (width, height) = (img_data_aug['width'], img_data_aug['height'])
                (rows, cols, _) = x_img.shape

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

                # resize the image so that smalles side is length = 600px
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)

                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height,
                                                     img_length_calc_function)
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image

                x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)
                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]
                x_img /= C.img_scaling_factor

                x_img = np.transpose(x_img, (2, 0, 1))
                x_img = np.expand_dims(x_img, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1] // 2:, :, :] *= C.std_scaling

                if backend == 'tf':
                    x_img = np.transpose(x_img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue
