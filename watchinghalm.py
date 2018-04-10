#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    :2018/3/31/0009 16:32
# @Author  : Wangkang
# @File    : watchinghalm.py
# @Software: PyCharm
# @(๑╹ヮ╹๑)ﾉ Studying makes me happy
import os, shutil, time, re
import win32file
import win32con

ACTIONS = {
    1: "bulid",
    2: "delete",
    3: "update",
    4: "Renamed from something",
    5: "Renamed to something"
}
FILE_LIST_DIRECTORY = 0x0001

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

# 离线EL watching_dir路径拼接
# 
#

#


# C:/halm/PVCTData/2018/201803/3-29D/EL_NG/2018_03_29_08
# print(path_to_watch)
# path_to_watch = 'C:/pythonProject/FRCNN/image/local_el'  

dst = '//172.16.10.231/Image/ELproject/my_el/'
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
        if action_type == 'bulid':
            #image_name = os.path.splitext(filename)[0]
            #new_name = image_name + '.jpg'
            dst_path = dst + filename
            print(dst_path)
            shutil.copy(full_filename, dst_path)
            #print('el_image was copyed~')
