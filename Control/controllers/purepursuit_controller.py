import numpy as np
import math


def Purepursuit(truck, nav_line):
    robot_state = np.zeros(2)  # 定义车辆位置
    robot_state[0] = truck.x
    robot_state[1] = truck.y+60  # 后轴中心位置
    dx, dy = np.average(nav_line[-12:], axis=0) - robot_state
    alpha = np.arctan(-dx/dy)  # dx, dy夹角
    ang = math.atan2(2.0 * truck.bev_l * np.sin(alpha), truck.ld) + 0.5  # pure pursuit controller
    return ang