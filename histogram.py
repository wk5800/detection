import cv2, tools, os


def histogram(dst, clahe_dir):
    """
    :param dst: 原始图片路径
    :param clahe_dir: 处理后的图片保存目录
    :return: 处理后的图片路径
    """

    img = cv2.imread(dst, 0)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(25, 25))
    cl1 = clahe.apply(img)
    file_name = dst.split('\\')[-1].split('.')[0]
    clahe_el_path = clahe_dir + file_name + '.jpg'

    cv2.imwrite(clahe_el_path, cl1)  # 把bmp\jpg格式经过处理后都转化成jpg格式

    return clahe_el_path


if __name__ == '__main__':
    for i in os.listdir('./image/test_el'):
        image = os.path.join('./image/test_el/', i)

