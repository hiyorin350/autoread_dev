import cv2
import numpy as np
from math import sqrt, atan2, pi
from set_vector import *

def process_img(image):

    def skeletonize(image):
        size = np.size(image)
        skel = np.zeros(image.shape, np.uint8)

        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(image, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(image, temp)
            skel = cv2.bitwise_or(skel, temp)
            image = eroded.copy()

            zeros = size - cv2.countNonZero(image)
            if zeros == size:
                break

        return skel

    image = cv2.imread('meter2.png')
    assert image is not None, "読み込みに失敗しました"
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #適応的二値化のパラメータ
    block_size = 11
    constant_c = 3

    # エッジ検出を用いて二値化 下記の適応的二値化とどっちが良いかは要検討
    # edges = cv2.Canny(gray, 130, 180, apertureSize=3)

    thres_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, constant_c)

    circles = cv2.HoughCircles(thres_image, cv2.HOUGH_GRADIENT, 1, 10, param1=250, param2=83, minRadius=20, maxRadius=85)

    s = []
    e = []
    l = []
    res = []

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]

            # 外側の塗りつぶし用のマスクを作成
            mask = np.zeros_like(gray)
            cv2.circle(mask, center, radius, 255, thickness=-1)

            # マスクを使って外側を白く塗りつぶす
            thres_image[mask == 0] = 255

        for (x, y, r) in circles[0, :]:
            # 円のトリミング、このタイイミングでcenterがずれている可能性あり
            # cropped_image = thres_image[y - int(0.9 * r):y + int(0.9 * r), x - int(0.9 * r):x + int(0.9 * r)]
            r = (int)(r * 0.9)
            cropped_image = thres_image[y - r:y + r, x - r:x + r]
            h, w = cropped_image.shape
            center_h = int(h / 2)
            center_w = int(w / 2)
            center = [center_w, center_h]
    else:
        print("円が検出されませんでした。")

    cropped_image = 255 - cropped_image

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cropped_image)
    max_width = 0
    max_width_label = -1
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if w > max_width:
            max_width = w
            max_width_label = label

    # 最大横幅領域を抽出
    max_width_region = np.zeros_like(cropped_image)
    max_width_region[labels == max_width_label] = 255

    skeleton = skeletonize(max_width_region)

    my_instance = DegreeHistgram()

    my_instance.histgramMethodUzGray(skeleton, center, s, e, l, res)

    result_image = skeleton

    return(result_image)
