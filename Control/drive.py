import pyvjoy
import numpy as np

global j
j = pyvjoy.VJoyDevice(1)


class Truck:  # 阿克曼转向模型
    def __init__(self):  # L:wheel base
        self.x = 400  # X
        self.y = 475  # Y
        self.theta = 0  # 车辆朝向与轨迹夹角
        self.dx = 0
        self.dy = 0
        self.dtheta = 0
        self.ang = 0
        self.acc = 0
        self.l = 6.0  # 轴距
        self.bev_l = 60  # bev下轴距
        self.speed = 0  # 速度
        self.bev_speed = 0  # bev下的速度
        self.dv = 0  # 速度变量
        self.L = 6.0  # 车辆全长
        self.k = 1.0
        self.ld = self.k * self.bev_speed + self.L  # 后轮到观察点距离
        self.dt = 0.2  # 决策间隔时间
        self.delta = 0   # 车辆与轨迹夹角

    def update(self, ang, acc, refer_time, speed):  # update ugv's state
        self.ang = ang
        self.acc = acc
        self.dv = speed - self.speed
        self.speed = speed
        self.bev_speed = self.speed / 3.6 * 6
        self.ld = self.k * self.bev_speed + self.bev_l
        self.dt = refer_time  # 更新推理时间
        self.dx = self.speed * np.cos(self.theta) * self.dt
        self.dy = self.speed * np.sin(self.theta) * self.dt
        self.dtheta = self.speed * np.tan(self.ang) / self.l * self.dt


def driver(ang, acc):
    j.data.wAxisX = int(ang * 32767)
    j.data.wAxisY = int(acc * 32767)
    j.data.wAxisZ = 0
    j.update()


def end():
    j.data.wAxisX = int(0.5 * 32767)
    j.data.wAxisY = int(0.5 * 32767)
    j.data.wAxisZ = 0
    j.update()
