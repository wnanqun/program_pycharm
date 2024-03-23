import cv2
import numpy as np

# 加载图像
images = ['AI.jpg', 'Fe.jpg', 'P.jpg']
# 每种化学元素在HSV空间中的颜色范围
colors = [
    {'lower': np.array([10, 0, 5]), 'upper': np.array([20, 255, 255])},
    {'lower': np.array([130, 0, 5]), 'upper': np.array([140, 255, 255])},
    {'lower': np.array([90, 0, 5]), 'upper': np.array([100, 255, 255])}
]

for img_file, color in zip(images, colors):
    # 读取图像
    img = cv2.imread(img_file)

    # 转换到HSV颜色空间（如果颜色在RGB空间中不容易区分）
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 定义颜色范围
    lower_color = np.array(color['lower'])
    upper_color = np.array(color['upper'])

    # 创建颜色掩膜
    mask = cv2.inRange(hsv_img, lower_color, upper_color)

    # 统计颜色掩膜中的非零像素数（即该颜色的像素数）
    count = cv2.countNonZero(mask)

    print(f"The number of elements in {img_file} is: {count}")