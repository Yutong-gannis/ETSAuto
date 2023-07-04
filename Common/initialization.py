import sys, os
import pycuda.autoinit

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))

sys.path.insert(0, os.path.abspath(project_path))
from Control.controllers.PID_controller import PID
from Condition.truck_condition import Truck

import numpy as np
import cv2


def init():
    truck = Truck()  # 初始化汽车状态模型
    tracks = [[1]]
    horizontal_pid = PID(3, 0.05, 0.01)  # 初始化横向PID控制算法
    vertical_pid = PID(0.03, 0, 0.1)  # 初始化纵向PID控制算法
    pts1 = np.array([[0, 0], [900, 0],
                     [0, 280], [900, 280]], dtype=np.float32)
    pts2 = np.array([[0, 0], [900, 0],
                     [350, 900], [550, 900]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)  # 原图的透视变换矩阵
    speed_limit = 0
    state = 'Cruise'
    nav_line = None
    return truck, tracks, horizontal_pid, vertical_pid, M, speed_limit, state, nav_line
