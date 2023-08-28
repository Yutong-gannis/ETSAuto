import numpy as np
import math


def Purepursuit(truck, nav_line):
    robot_state = np.zeros(2)  # 定义车辆位置
    robot_state[1] = truck['x']
    robot_state[0] = truck['y'] - truck['wheelbase']  # 后轴中心位置
    dy, dx = np.average(nav_line[30:40], axis=0) - robot_state
    alpha = np.arctan(dx / dy)  # dx, dy夹角
    ang = math.atan2(2.0 * truck['wheelbase'] * np.sin(alpha), truck['ld'])  # pure pursuit controller
    return ang
