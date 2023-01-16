import cv2
import numpy as np

class Stop_Line:  # 对停止线进行封装
    def __init__(self, center_y):
        self.center_y = center_y
        self.length = 80

def line_filter(img, M):
    img = img[400:680, 100:1000, :]
    img = cv2.warpPerspective(img, M, (900, 900))
    img = cv2.resize(img, (300, 300))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    img = cv2.erode(img, kernel, iterations=1)

    img = cv2.Sobel(img, -1, 0, 2, ksize=3)
    img = cv2.Canny(img, 250, 300)
    img = cv2.dilate(img, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img = cv2.erode(img, kernel, iterations=1)

    stop_line = detect_stop_line(img, 0.15)

    return stop_line

def detect_stop_line(img, conf):
    sum_y = np.average(img, axis=1) / 255
    idxs = np.where(sum_y >= conf)[0]
    if len(idxs.tolist()) >= 1:
        stop_line_pos = 300 - np.average(idxs)
        stop_line = Stop_Line(stop_line_pos/4)
    else:
        stop_line = None
    return stop_line