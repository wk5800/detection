#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/3/31/0009 16:32
# @Author  : Wangkang
# @File    : 文件监控离线.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy

import os, shutil, time, re
import win32file
import win32con
import threading

ACTIONS = {
    1: "build",
    2: "delete",
    3: "update",
    4: "Renamed from something",
    5: "Renamed to something"
}

FILE_LIST_DIRECTORY = 0x0001
'''
# halm在线EL  path_to_watch的拼接
father_dir = 'C:/halm/PVCTData'
ticks = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
date = time.localtime(time.time())

year = str(date.tm_year)
month = str(date.tm_mon)
day = str(date.tm_mday)
hour = str(date.tm_hour)

ticks_month = ticks.split('-')[1] # 01~12
ticks_day = ticks.split(' ')[0].split('-')[2] # 01~30
ticks_hour = ticks.split(' ')[1].split(':')[0] # 01~24
dir1 = year  # 2018
dir2 = year + ticks_month # 201803
dir3 = os.path.join(dir1, dir2)  # 2018/201803

if int(hour) >= 7 and int(hour) <= 20:
    dir4 = month + '-' + day + 'D'  # 3-29D
else:
    dir4 = month + '-' + day + 'N'  # 3-29N

dir5 = os.path.join(dir3, dir4)  # 2018/201803/3-29D
dir6 = os.path.join(dir5, 'EL_NG')  # 2018/201803/3-29D/EL_NG
dir7 = year + '_' + ticks_month + '_' + ticks_day + '_' + ticks_hour  # 2018_03_29_08

el_path = os.path.join(dir6, dir7)  # 2018/201803/3-29D/EL_NG/2018_03_29_08
path_to_watch = os.path.join(father_dir, el_path)  # C:/halm/PVCTData/2018/201803/3-29D/EL_NG/2018_03_29_08

'''

# 离线EL watching_dir路径拼接


while True:
    """
    每隔一段时间，重新载入监控目录，对指定目录进行重新监控，复制等操作~~~
    """
    date = time.localtime(time.time())
    year = str(date.tm_year)
    month = str(date.tm_mon)
    day = str(date.tm_mday)
    hour = str(date.tm_hour)
    chinesemonth = ['一月份', '二月份', '三月份', '四月份', '五月份', '六月份', '七月份', '八月份', '九月份', '十月份', '十一月份', '十二月份']
    ROOT_dir = 'D:\%s' % chinesemonth[date.tm_mon - 1]

    if date.tm_hour >= 7 and date.tm_hour <= 20:
        DATE_dir = year + '-' + month + '-' + day + 'D'  # 2018-3-29D
    else:
        DATE_dir = year + '-' + month + '-' + day + 'N'  # 2018-3-29N

    Offline_father_dir = os.path.join(ROOT_dir, DATE_dir)  # 可能是真正的离线EL father_dir

    # 2018年3月30日 father_dir 测试~~~~~~~~~~~~~~
    father_dir330 = '//192.168.30.3/Users/lenovo/Desktop'  # 测试版 开头
    #father_dir330 = '..'
    father_dir = 'C:/halm/PVCTData/2018'  # 需要开头

    dst = '//172.16.10.231/Image/ELproject/my_el/'  # 目标路径
    dst330 = 'D:/Image/ELproject/my_el/'  # 测试版 目标路径

    watching_path_list = []
    for root, dirs, files in os.walk(father_dir330, topdown=False):
        # for name in files:
        # print('文件路径: ', os.path.join(root, name))
        for name in dirs:
            sub = os.path.join(root, name)  # 输出所有father_dir下的目录
            last_dir = sub.split('\\')[-1]  # 输出所有路径下最后一个文件夹名

            # 需要结尾
            if re.search('ng', last_dir, re.I):  # 匹配最后一个目录名是否包含‘NG’字段(不分大小写)
                watching_path_list.append(sub)


    def start_monitor(path_to_watch):
        print('watching dir：', path_to_watch)
        hDir = win32file.CreateFile(
            path_to_watch,
            FILE_LIST_DIRECTORY,
            win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
            None,
            win32con.OPEN_EXISTING,
            win32con.FILE_FLAG_BACKUP_SEMANTICS,
            None
        )
        while True:
            results = win32file.ReadDirectoryChangesW(
                hDir,
                1024,
                True,
                win32con.FILE_NOTIFY_CHANGE_FILE_NAME |
                win32con.FILE_NOTIFY_CHANGE_DIR_NAME |
                win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES |
                win32con.FILE_NOTIFY_CHANGE_SIZE |
                win32con.FILE_NOTIFY_CHANGE_LAST_WRITE |
                win32con.FILE_NOTIFY_CHANGE_SECURITY,
                None,
                None)
            for action, filename in results:
                full_filename = os.path.join(path_to_watch, filename)
                action_type = ACTIONS.get(action, "Unknown")
                if action_type == 'build':
                    new_filename = os.path.join(path_to_watch, filename)
                    print('产生新图片： ', new_filename)
                    dst_path = dst330 + filename  # 新产生的图片文件移动路径
                    print('复制到: ', dst_path)
                    try:
                        shutil.copy(full_filename, dst_path)
                        # time.sleep(0.5)
                    except FileNotFoundError:
                        continue


    for path in watching_path_list:
        monitor_thread = threading.Thread(target=start_monitor, args=(path,))
        monitor_thread.start()
    time.sleep(1800)  # 半小时刷新一次监控目录

