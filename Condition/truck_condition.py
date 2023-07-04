import numpy as np
import math


class Truck:  # 阿克曼转向模型
    def __init__(self):  # L:wheel base
        self.x = 0  # X
        self.y = 0  # Y
        self.theta = 0  # 车辆朝向与轨迹夹角
        self.dx = 0
        self.dy = 0
        self.dtheta = 0
        self.ang = 0
        self.acc = 0
        self.wheelbase = 12.0  # 轴距
        self.centerToFront = 5.0
        self.mass = 5
        self.tireStiffnessRear = 5
        self.ld = 20  # 预瞄距离
        self.lf = 12  # 预瞄距离补偿
        self.speed = 0  # 速度
        self.bev_speed = 0  # bev下的速度
        self.dv = 0  # 速度变量
        self.dt = 0.15  # 决策间隔时间
        self.delta = 0   # 车辆与轨迹夹角

    def update(self, ang, acc, refer_time):  # update ugv's state
        self.ang = ang
        self.acc = acc
        self.bev_speed = self.speed / 3.6 * 6
        self.ld = 1.0 * self.bev_speed + self.lf
        self.dt = refer_time  # 更新推理时间
        self.dx = self.speed * np.cos(self.theta) * self.dt
        self.dy = self.speed * np.sin(self.theta) * self.dt
        self.dtheta = self.speed * np.tan(self.ang) / self.wheelbase * self.dt

    def state_space(self, ref_delta, ref_yaw):  # 将模型离散化后的状态空间表达
        A = np.matrix([
            [1.0, 0.0, -self.speed * self.dt * math.sin(ref_yaw)],
            [0.0, 1.0, self.speed * self.dt * math.cos(ref_yaw)],
            [0.0, 0.0, 1.0]])

        B = np.matrix([
            [self.dt * math.cos(ref_yaw), 0],
            [self.dt * math.sin(ref_yaw), 0],
            [self.dt * math.tan(ref_delta) / self.wheelbase,
             self.speed * self.dt / (self.wheelbase * math.cos(ref_delta) * math.cos(ref_delta))]
        ])

        C = np.eye(3)
        return A, B, C
