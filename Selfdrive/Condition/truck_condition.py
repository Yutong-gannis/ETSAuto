import numpy as np
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Message.iodata import save_pkl
from lib.ets2sdktelemetry import Ets2SdkTelemetry
from lib.sharedmemory import SharedMemory


class Truck:  # 阿克曼转向模型
    """This is class record the condition of truck"""
    def __init__(self):  # L:wheel base
        self.coordinateX = 0  # X
        self.coordinateY = 0  # Y
        self.theta = 0  # 车辆朝向与轨迹夹角
        self.dX = 0
        self.dY = 0
        self.dtheta = 0
        self.ang = 0
        self.acc = 0
        self.wheelbase = 8.0  # 轴距
        self.ld = 20  # 预瞄距离
        self.lf = 3.63  # 预瞄距离补偿
        self.speed = 0  # 速度
        self.speedlimit = None #
        self.overspeed = False
        self.dt = 0.04  # 间隔时间
        self.telemetry = Ets2SdkTelemetry()
        self.sharemem = SharedMemory()

    def update(self):
        """This is function update the truck's condition
        """
        data = self.sharemem.update()
        print(data.lightsDashboard)

        self.dtheta = self.theta - data.rotationX * 2 * np.pi  # 逆时针为正
        self.theta = data.rotationX * 2 * np.pi
        self.speed = data.speed
        self.speedlimit = data.speedlimit
        
        self.overspeed = self.speed > self.speedlimit
        self.acc = [data.accelerationX, data.accelerationY]
        self.dX = self.speed * np.cos(self.dtheta) * self.dt
        self.dY = self.speed * np.sin(self.dtheta) * self.dt

    def publish(self):
        """This is function to publish condition data
        """
        condition_dict = {'x': self.coordinateX,
                          'y': self.coordinateY,
                          'theta': self.theta,
                          'dx': self.dX,
                          'dy': self.dY,
                          'dtheta': self.dtheta,
                          'ang': self.ang,
                          'acc': self.acc,
                          'wheelbase': self.wheelbase,
                          'ld': self.ld,
                          'lf': self.lf,
                          'speed': self.speed,
                          'speedlimit': self.speedlimit,
                          'overspeed': self.overspeed,
                          'dt': self.dt}
        save_pkl(os.path.join(project_path, 'Message/temp/condition.pkl'), condition_dict)

