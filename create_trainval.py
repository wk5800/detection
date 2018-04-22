#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/4/22/0022 14:17
# @Author  : Wangkang
# @File    : create_trainval.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/4/20/0020 10:20
# @Author  : Wangkang
# @File    : create_trainval.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy


import os

trainpath = 'C:/pythonProject/keras-frcnn-master/train_data/VOCdevkit/VOC2012/'

xmlpath = os.path.join(trainpath, 'Annotations')
trainval = os.path.join(trainpath, 'ImageSets/Main/trainval.txt')
print(trainval)
if os.path.isfile(trainval) is False:
    f = open(trainval, 'a')
else :
    f = open(trainval, 'a')
    for xml in os.listdir(xmlpath):

        xmlname = os.path.splitext(xml)[0]
        print(xmlname)
        f.write(xmlname+'\n')  # 这里的\n的意思是在源文件末尾换行，即新加内容另起一行插入。
    f.close()



