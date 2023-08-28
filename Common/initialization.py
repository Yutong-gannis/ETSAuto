import sys, os
import numpy as np
import cv2

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))

sys.path.insert(0, os.path.abspath(project_path))
from Condition.truck_condition import Truck


def init():
    truck = Truck()  # 初始化汽车状态模型
    tracks = [[1]]
    pts1 = np.array([[0, 0], [900, 0],
                     [0, 280], [900, 280]], dtype=np.float32)
    pts2 = np.array([[0, 0], [900, 0],
                     [350, 900], [550, 900]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)  # 原图的透视变换矩阵
    speed_limit = 0
    state = 'Cruise'
    nav_line = None
    return truck, tracks, M, speed_limit, state, nav_line
