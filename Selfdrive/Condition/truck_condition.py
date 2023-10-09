import numpy as np
import os
import sys
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
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
        condition_dict_pub = SharedMemoryDict(name='condition', size=1024)
        condition_dict_pub['x'] =  self.coordinateX
        condition_dict_pub['y'] =  self.coordinateY
        condition_dict_pub['theta'] =  self.theta
        condition_dict_pub['dx'] =  self.dX
        condition_dict_pub['dy'] =  self.dY
        condition_dict_pub['dtheta'] =  self.dtheta
        condition_dict_pub['ang'] =  self.ang
        condition_dict_pub['acc'] =  self.acc
        condition_dict_pub['wheelbase'] =  self.wheelbase
        condition_dict_pub['ld'] =  self.ld
        condition_dict_pub['lf'] =  self.lf
        condition_dict_pub['speed'] =  self.speed
        condition_dict_pub['speedlimit'] =  self.speedlimit
        condition_dict_pub['overspeed'] =  self.overspeed
        condition_dict_pub['dt'] =  self.dt
