import os, re

father_dir = 'C:/halm/PVCTData/2018'  # 需要开头
watching_path_list = []
for root, dirs, files in os.walk(father_dir, topdown=False):
    # for name in files:
    # print('文件路径: ', os.path.join(root, name))
    for name in dirs:
        sub = os.path.join(root, name)  # 输出所有father_dir下的目录
        last_dir = sub.split('\\')[-1]  # 输出所有路径下最后一个文件夹名

        # 需要结尾
        if re.search('ng', last_dir, re.I):  # 匹配最后一个目录名是否包含‘NG’字段(不分大小写)
            watching_path_list.append(sub)

print(watching_path_list)

import win32file
import tempfile
import threading
import win32con

dirs = ["C:\\WINDOWS\\TEMP", tempfile.gettempdir()]


def start_monitor(path_to_watch):
    h_directory = win32file.CreateFile(path_to_watch, win32con.GENERIC_READ,
                                       win32con.FILE_SHARE_DELETE | win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
                                       None, win32con.OPEN_EXISTING, win32con.FILE_FLAG_BACKUP_SEMANTICS, None)
    while True:
        try:
            results = win32file.ReadDirectoryChangesW(h_directory, 1024, True,
                                                      win32con.FILE_NOTIFY_CHANGE_FILE_NAME | win32con.FILE_NOTIFY_CHANGE_DIR_NAME | win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES | win32con.FILE_NOTIFY_CHANGE_SIZE | win32con.FILE_NOTIFY_CHANGE_LAST_WRITE | win32con.FILE_NOTIFY_CHANGE_SECURITY,
                                                      None)
            for action, filename in results:
                print(action)
                print(filename)
        except:
            pass


for path in dirs:
    monitor_thread = threading.Thread(target=start_monitor, args=(path,))
    monitor_thread.start()
