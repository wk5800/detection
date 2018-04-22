"""
然后搭建网络，载入权重，传递参数，计算mAP～～
在train_frcnn中，保存了配置信息config.pickle，后续分类网络送入1：1的正负样本
"""
from __future__ import division
import random
import pprint
import sys, os
import time
import numpy as np
from optparse import OptionParser
import pickle

from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras_frcnn import config, data_generators
from keras_frcnn import losses as losses
import keras_frcnn.roi_helpers as roi_helpers
from keras.utils import generic_utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.setrecursionlimit(40000)

parser = OptionParser()

# 训练数据的路径
parser.add_option("-p", "--path", dest="train_path", help="Path to training data.", default='C:/pythonProject/keras-frcnn-master/train_data/VOCdevkit')
# 解析xml
parser.add_option("-o", "--parser", dest="parser", help="Parser to use. One of simple or pascal_voc",
                  default="pascal_voc")
# 每次处理的rois数量
parser.add_option("-n", "--num_rois", type="int", dest="num_rois", help="Number of RoIs to process at once.",
                  default=10)
# 网络层选择
parser.add_option("--network", dest="network", help="Base network to use. Supports vgg or resnet50.",
                  default='resnet50')
# 水平翻转
parser.add_option("--hf", dest="horizontal_flips", help="Augment with horizontal flips in training. (Default=false).",
                  action="store_true", default=True)
# 垂直翻转
parser.add_option("--vf", dest="vertical_flips", help="Augment with vertical flips in training. (Default=false).",
                  action="store_true", default=False)
# 图片翻转90度
parser.add_option("--rot", "--rot_90", dest="rot_90",
                  help="Augment with 90 degree rotations in training. (Default=false).",
                  action="store_true", default=True)
# epoch次数
parser.add_option("--num_epochs", type="int", dest="num_epochs", help="Number of epochs.", default=20)
# 配置文件保存路径
parser.add_option("--config_filename", dest="config_filename", help="Location to store all the metadata related to the training (to be used when testing).",
                  default="C:/pythonProject/keras-frcnn-master/model/config.pickle")
# 模型保存路径
parser.add_option("--output_weight_path", dest="output_weight_path", help="Output path for weights.",
                  default='C:/pythonProject/keras-frcnn-master/model/model_frcnn.hdf5')
# 网络初始权重路径
parser.add_option("--input_weight_path", dest="input_weight_path",
                  help="Input path for weights. If not specified, will try to load default weights provided by keras.",
                  default='C:/pythonProject/keras-frcnn-master/model/resnet50_weights_tf_dim_ordering_tf_kernels.h5')

(options, args) = parser.parse_args()

if not options.train_path:  # if filename is not given
    parser.error('Error: path to training data must be specified. Pass --path to command line')

if options.parser == 'pascal_voc':
    from keras_frcnn.pascal_voc_parser import get_data
elif options.parser == 'simple':
    from keras_frcnn.simple_parser import get_data
else:
    raise ValueError("Command line option parser must be one of 'pascal_voc' or 'simple'")

# pass the settings from the command line, and persist them in the config object
C = config.Config()

C.use_horizontal_flips = bool(options.horizontal_flips)
C.use_vertical_flips = bool(options.vertical_flips)
C.rot_90 = bool(options.rot_90)

C.model_path = options.output_weight_path
C.num_rois = int(options.num_rois)  # 每次取的感兴趣区域，默认为32（C.num_rois决定了网络一次最多可以探测多少个物体）


if options.network == 'vgg':
    C.network = 'vgg'
    from keras_frcnn import vgg as nn
elif options.network == 'resnet50':
    from keras_frcnn import resnet as nn

    C.network = 'resnet50'
else:
    print('Not a valid model')
    raise ValueError

# check if weight path was passed via command line
if options.input_weight_path:
    C.base_net_weights = options.input_weight_path
else:
    # set the path to weights based on backend and model
    C.base_net_weights = nn.get_weight_path()

all_imgs, classes_count, class_mapping = get_data(options.train_path)


if 'bg' not in classes_count:
    classes_count['bg'] = 0

    # 读取每一个类对应的数字（如：'person': 0）。如果class_mapping中没有背景类的话，添加背景类且其对应的数字为len(class_mapping)
    class_mapping['bg'] = len(class_mapping)

C.class_mapping = class_mapping

# 类别对应数字的信息以字典形式存储，其中v是数字，k是类别
inv_map = {v: k for k, v in class_mapping.items()}

print('Training images per class:')
pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))

config_output_filename = options.config_filename  # 配置文件地址

with open(config_output_filename, 'wb') as config_f:
    pickle.dump(C, config_f)
    print('Config has been written to {}, and can be loaded when testing to ensure correct results'.format(
        config_output_filename))

random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'val']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, nn.get_img_output_length, K.image_dim_ordering(), mode='val')

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(None, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)  # feature map
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

# 构建rpn输出，返回[x_class, x_regr, base_layers]
rpn = nn.rpn(shared_layers, num_anchors)


# 构建classifier输出，输入参数分别是：特征层输出，预选框，探测框的数目，多少个类，是否可训练
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

model_rpn = Model(img_input, rpn[:2])
model_classifier = Model([img_input, roi_input], classifier)

# 这是一个既包含RPN又包含分类器的模型，用于为模型加载/保存权重。
model_all = Model([img_input, roi_input], rpn[:2] + classifier)

try:
    print('loading weights from {}'.format(C.base_net_weights))
    model_rpn.load_weights(C.base_net_weights, by_name=True)
    model_classifier.load_weights(C.base_net_weights, by_name=True)
except:
    print('Could not load pretrained model weights. Weights can be found in the keras application folder \
		https://github.com/fchollet/keras/tree/master/keras/applications')

optimizer = Adam(lr=1e-5)
optimizer_classifier = Adam(lr=1e-5)
model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
model_classifier.compile(optimizer=optimizer_classifier,
                         loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count) - 1)],
                         metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
model_all.compile(optimizer='sgd', loss='mae')

epoch_length = 2500

num_epochs = int(options.num_epochs)
iter_num = 0

losses = np.zeros((epoch_length, 5))
rpn_accuracy_rpn_monitor = []
rpn_accuracy_for_epoch = []
start_time = time.time()

best_loss = np.Inf

class_mapping_inv = {v: k for k, v in class_mapping.items()}
print('Starting training')

vis = True

for epoch_num in range(num_epochs):

    #  Progbar是生成进度条(这是一个武大的兄弟告诉我的，表示感谢)
    progbar = generic_utils.Progbar(epoch_length)
    print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))

    while True:
        try:

            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = []
                print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                    mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print(
                        'RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

            # data_gen_train是一个迭代器。返回的是 np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug（我们这里假设数据没有进行水平翻转等操作。
            # 那么，x_img = img_data_aug）,y_rpn_cls和y_rpn_regr是RPN的两个损失函数。
            # 训练RPN网络,X是图片、Y是对应类别和回归梯度
            X, Y, img_data = next(data_gen_train)

            loss_rpn = model_rpn.train_on_batch(X, Y)

            P_rpn = model_rpn.predict_on_batch(X)  # 最终的预测


            # rpn_to_roi接收的是每张图片的预测输出，返回的R = [boxes, probs]
            R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # 通过calc_iou()找出剩下的不多的region对应ground truth里重合度最高的bbox，从而获得model_classifier的数据和标签。
            # X2保留所有的背景和match bbox的框； Y1 是类别one-hot转码；
            # Y2是对应类别的标签及回归要学习的坐标位置; IouS是debug用的。
            X2, Y1, Y2, IouS = roi_helpers.calc_iou(R, img_data, C, class_mapping)

            if X2 is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            # 得到正负样本在的位置。（Y1[0, :, -1]中，0指定batch的位置，：指所有框，-1指最后一个维度即背景类）
            neg_samples = np.where(Y1[0, :, -1] == 1)
            pos_samples = np.where(Y1[0, :, -1] == 0)

            if len(neg_samples) > 0:
                neg_samples = neg_samples[0]  # 将其变为一维的数组
            else:
                neg_samples = []

            if len(pos_samples) > 0:
                pos_samples = pos_samples[0]  # 将其变为一维的数组
            else:
                pos_samples = []

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                """选择C.num_rois个数的框，送入classifier网络进行训练"""

                if len(pos_samples) < C.num_rois // 2:
                    """当正样本个数<32/2 时，正样本全部保留作为训练样本"""
                    selected_pos_samples = pos_samples.tolist()
                else:
                    """正样本随机选择32/2个作为训练样本"""
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()
                try:
                    """负样本为32-正样本的个数"""
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=False).tolist()
                except:
                    selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                            replace=True).tolist()

                sel_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    sel_samples = random.choice(neg_samples)
                else:
                    sel_samples = random.choice(pos_samples)

            loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

            losses[iter_num, 0] = loss_rpn[1]
            losses[iter_num, 1] = loss_rpn[2]

            losses[iter_num, 2] = loss_class[1]
            losses[iter_num, 3] = loss_class[2]
            losses[iter_num, 4] = loss_class[3]

            iter_num += 1

            progbar.update(iter_num,
                           [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                            ('detector_cls', np.mean(losses[:iter_num, 2])),
                            ('detector_regr', np.mean(losses[:iter_num, 3]))])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                # 每一次训练平均提供的可用正样本回归框的个数
                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = []

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                # 得到这一轮总的平均loss
                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    """当结束一轮的epoch时，只有当这轮epoch的loss小于最优的时候才会存储这轮的训练数据。并结束这轮epoch进入下一轮epoch"""
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('Training complete, exiting.')
