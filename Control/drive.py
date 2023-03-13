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
        self.lf = 0  # 预瞄距离补偿
        # self.lf_temp = 0.5
        self.speed = 0  # 速度
        self.bev_speed = 0  # bev下的速度
        self.dv = 0  # 速度变量
        self.dt = 0.2  # 决策间隔时间
        self.delta = 0   # 车辆与轨迹夹角
        # self.speed_limits = []  # 连续记录八次速度限制
        # self.speed_limits_slope = 0
        # self.d_speed_limits_slope = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def update(self, ang, acc, refer_time, speed, info):  # update ugv's state
        self.ang = ang
        self.acc = acc
        self.dv = speed - self.speed
        self.speed = speed
        self.bev_speed = self.speed / 3.6 * 6

        # if len(self.speed_limits) < 8:
        #     self.speed_limits.append(speed_limit)
        # elif len(self.speed_limits) == 8:
        #     self.speed_limits.pop(0)
        #     self.speed_limits.append(speed_limit)
        #     speed_limits = np.array(self.speed_limits)
        #     t = np.arange(len(speed_limits))
        #     p = np.polyfit(t, speed_limits, 1)
        #     self.d_speed_limits_slope.pop(0)
        #     self.d_speed_limits_slope.append(p[0] - self.speed_limits_slope)
        #     self.speed_limits_slope = p[0]

        # if self.d_speed_limits_slope[-1] < 0:
        #     if self.lf_temp > 0:
        #         self.lf_temp = -0.5
        #     if self.lf_temp < -1:
        #         self.lf_temp = -0.9
        #     else:
        #         self.lf_temp = self.lf_temp - (self.lf_temp + 0.29) ** 2
        #     if self.lf_temp > -1 and self.speed < 65:
        #         self.lf = self.lf + self.lf_temp * 100 + 50
        # elif self.d_speed_limits_slope[-1] > 0:
        #     if self.lf_temp < 0:
        #         self.lf_temp = 0.5
        #     if self.lf_temp > 1:
        #         self.lf_temp = 0.9
        #     else:
        #         self.lf_temp = self.lf_temp + (self.lf_temp - 0.29) ** 2
        #     if self.lf_temp < 1 and self.speed < 65:
        #         self.lf = self.lf + self.lf_temp * 100 - 50

        # if self.lf < 20:
        #     self.lf = 20
        # elif self.lf > 110:
        #     self.lf = 110

        # if len(info.direction) > 0:
        #     self.lf = 150
        # else:
        #     self.lf = 0
        # info.direction = []

        self.ld = 1.0 * self.bev_speed + 160 + self.lf

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
        self.direction = []  # 转向信息(大于0为左转，小于0为右转)

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
