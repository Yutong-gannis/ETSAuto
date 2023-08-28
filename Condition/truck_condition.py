import numpy as np
import pickle


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
        self.wheelbase = 8.0  # 轴距
        self.ld = 20  # 预瞄距离
        self.lf = 10  # 预瞄距离补偿
        self.speed = 0  # 速度
        self.dt = 0.15  # 决策间隔时间

    def update(self, ang, acc, refer_time):  # update ugv's state
        self.ang = ang
        self.acc = acc
        self.ld = 2 * self.speed + self.lf
        self.dt = refer_time  # 更新推理时间
        self.dx = self.speed * np.cos(self.theta) * self.dt
        self.dy = self.speed * np.sin(self.theta) * self.dt
        self.dtheta = self.speed * np.tan(self.ang) / self.wheelbase * self.dt

    def publish(self):
        condition_list = {'x': self.x,
                          'y': self.y,
                          'theta': self.theta,
                          'dx': self.dx,
                          'dy': self.dy,
                          'dtheta': self.dtheta,
                          'ang': self.ang,
                          'acc': self.acc,
                          'wheelbase': self.wheelbase,
                          'ld': self.ld,
                          'lf': self.lf,
                          'speed': self.speed,
                          'dt': self.dt}
        condition_file = open('temp/condition.pkl', 'wb')
        pickle.dump(condition_list, condition_file)
        condition_file.close()

