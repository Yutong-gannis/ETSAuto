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
        self.l = 12.0  # 轴距
        self.bev_l = 120  # bev下轴距
        self.ld = 120  # 预瞄距离
        self.lf = 160  # 预瞄距离补偿
        self.speed = 0  # 速度
        self.bev_speed = 0  # bev下的速度
        self.dv = 0  # 速度变量
        self.dt = 0.2  # 决策间隔时间
        self.delta = 0   # 车辆与轨迹夹角

    def update(self, ang, acc, refer_time, speed, info):  # update ugv's state
        self.ang = ang
        self.acc = acc
        self.dv = speed - self.speed
        self.speed = speed
        self.bev_speed = self.speed / 3.6 * 6
        self.ld = 1.0 * self.bev_speed + self.lf
        self.dt = refer_time  # 更新推理时间
        self.dx = self.speed * np.cos(self.theta) * self.dt
        self.dy = self.speed * np.sin(self.theta) * self.dt
        self.dtheta = self.speed * np.tan(self.ang) / self.l * self.dt


class Info:
    def __init__(self):
        self.activeAP = False  # 是否激活自动驾驶
        self.roads_type = 0  # 0：普通道路 1：高速公路
        self.road_speed = [50, 50, 45, 40, 30]  # 道路速度
        self.AP_exit_reason = 0  # 自动驾驶退出原因 0：正常退出 1：地图导航获取路线出错
        self.direction = 0  # -1：左转 0：直行 1：右转

    def update(self, roads_type):  # 更新道路类型
        self.roads_type = roads_type
        self.AP_exit_reason = 0
        if self.roads_type == 0:
            self.road_speed[0] = 50
            self.road_speed[1] = 50
            self.road_speed[2] = 45
            self.road_speed[3] = 40
            self.road_speed[4] = 30
        elif self.roads_type == 1:
            self.road_speed[0] = 90
            self.road_speed[1] = 90
            self.road_speed[2] = 80
            self.road_speed[3] = 70
            self.road_speed[4] = 60


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
