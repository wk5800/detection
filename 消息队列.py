import queue, gc
import threading
import time, histogram
import os, shutil, time
import win32file
import win32con


ACTIONS = {
    1: "创建",
    2: "删除",
    3: "更新",
    4: "Renamed from something",
    5: "Renamed to something"
}

FILE_LIST_DIRECTORY = 0x0001
dir_to_watch = 'C:/pythonProject/FRCNN/image/my_el'
clahe_dir = 'C:/pythonProject/FRCNN/image/clahe_el/'
predict_el_dir = 'C:/pythonProject/FRCNN/image/predict_el/'
processed_el_dir = 'C:/pythonProject/FRCNN/image/processed_el/'


hDir = win32file.CreateFile(
    dir_to_watch,
    FILE_LIST_DIRECTORY,
    win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
    None,
    win32con.OPEN_EXISTING,
    win32con.FILE_FLAG_BACKUP_SEMANTICS,
    None
)

new_queue = queue.Queue(maxsize=5)  # 创建一个queue.Queue() 的实例


class ThreadNum(threading.Thread):

    def __init__(self, queue):
        threading.Thread.__init__(self)  # 将经过填充数据的实例传递给线程类，后者是通过继承 threading.Thread 的方式创建的
        self.queue = queue

    def run(self):
        while True:
            # 消费者端
            num = self.queue.get()  # 从队列中获取num元素
            print('需计算的图片id: %s' % num)
            time.sleep(2)  # 目的是为了让图片完整的保存下来
            clahe_el_path = histogram.histogram(dst=num, clahe_dir=clahe_dir)
            shutil.move(num, processed_el_dir)
            gc.collect()
            self.queue.task_done()  # 在完成这项工作之后，使用 queue.task_done() 函数向任务已经完成的队列发送一个信号，用于表示队列中的某个元素已执行完成，该方法会被下面的join（）使用


start = time.time()


def main():
    # 产生一个 threads pool 守护线程池。, 并把消息传递给thread函数进行处理，这里开启10个并发。
    for i in range(1):
        t = ThreadNum(new_queue)  # 将经过填充数据的实例传递给线程类，后者是通过继承 threading.Thread 的方式创建的
        t.setDaemon(True)  # 通过将守护线程设置为 true，程序运行完自动退出。
        t.start()

    # 生产者端
    while True:
        results = win32file.ReadDirectoryChangesW(
            hDir, 1024, True,
            win32con.FILE_NOTIFY_CHANGE_FILE_NAME |
            win32con.FILE_NOTIFY_CHANGE_DIR_NAME |
            win32con.FILE_NOTIFY_CHANGE_ATTRIBUTES |
            win32con.FILE_NOTIFY_CHANGE_SIZE |
            win32con.FILE_NOTIFY_CHANGE_LAST_WRITE |
            win32con.FILE_NOTIFY_CHANGE_SECURITY,
            None,
            None)
        for action, filename in results:
            full_filename = os.path.join(dir_to_watch, filename)
            action_type = ACTIONS.get(action, "Unknown")  # 操作形式
            if action_type == '创建':
                new_queue.put(full_filename)
                new_queue.join()





if __name__ == '__main__':

    main()
    print("Elapsed Time: %s" % (time.time() - start))
