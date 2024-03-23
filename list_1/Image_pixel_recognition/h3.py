import cv2
import numpy as np

# 加载图像
image_AI = cv2.imread("AI.jpg")
image_Fe = cv2.imread("Fe.jpg")
image_P = cv2.imread("P.jpg")

# 将图像转换为HSV颜色空间
image_AI_hsv = cv2.cvtColor(image_AI, cv2.COLOR_BGR2HSV)
image_Fe_hsv = cv2.cvtColor(image_Fe, cv2.COLOR_BGR2HSV)
image_P_hsv = cv2.cvtColor(image_P, cv2.COLOR_BGR2HSV)

# 定义颜色范围
color_ranges = {
    "AI": {
        "lower": np.array([10, 0, 5]),
        "upper": np.array([20, 255, 255]),
    },
    "Fe": {
        "lower": np.array([130, 0, 5]),
        "upper": np.array([140, 255, 255]),
    },
    "P": {
        "lower": np.array([90, 0, 5]),
        "upper": np.array([100, 255, 255]),
    },
}

# 创建掩码函数
def create_mask(image, color_range):
    mask = cv2.inRange(image, color_range["lower"], color_range["upper"])
    return mask

# 创建重叠图像函数
def create_overlap_image(mask1, mask2, mask3):
    tmp = cv2.bitwise_and(mask1, mask2)
    overlap = cv2.bitwise_and(mask3, tmp)

    return overlap

# 统计重叠像素数函数
def count_overlapping_pixels(overlap):
    count = cv2.countNonZero(overlap)
    return count

# 统计三种元素重叠情况
overlap_AI_Fe_P = create_overlap_image(create_mask(image_AI_hsv, color_ranges["AI"]),
                                        create_mask(image_Fe_hsv, color_ranges["Fe"]),
                                        create_mask(image_P_hsv, color_ranges["P"]))

# 统计重叠像素数
count_AI_Fe_P = count_overlapping_pixels(overlap_AI_Fe_P)

# 打印结果
print("AI, Fe, and P overlapping pixels:", count_AI_Fe_P)

# 展示重叠情况
cv2.imshow("AI, Fe, and P overlap", overlap_AI_Fe_P)
cv2.waitKey(0)
cv2.destroyAllWindows()
