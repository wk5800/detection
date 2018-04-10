#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/3/22/0022 14:54
# @Author  : Wangkang
# @File    : 获取my_el文件创建时间.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy
import os

my_el = 'C:/pythonProject/FRCNN/image/my_el'
for i in os.listdir(my_el):
    name = os.path.join(my_el,i)
    create_time = os.path.getctime(name) #获取文件的创建时间
    print('%s的创建时间：%s' % (name,create_time))