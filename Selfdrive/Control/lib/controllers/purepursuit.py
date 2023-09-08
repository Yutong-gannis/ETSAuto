import numpy as np
import math


class PurePursuit:
    def __init__(self):
        self.ld = 20  # 预瞄距离
        self.lf = 3.63  # 预瞄距离补偿
        self.wheelbase = 8.0  # 轴距
        
    def run(self, trajectory, speed):
        self.ld = 1.0 * speed + self.lf
        robot_state = np.zeros(2)  # 定义车辆位置
        robot_state[1] = 0
        robot_state[0] = 0 - self.wheelbase  # 后轴中心位置
        dy, dx = np.average(trajectory[30:40], axis=0) - robot_state
        alpha = np.arctan(dx / dy)  # dx, dy夹角
        ang = math.atan2(2.0 * self.wheelbase * np.sin(alpha), self.ld)  # pure pursuit controller
        return ang
