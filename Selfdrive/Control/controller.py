import os
import numpy as np
import sys
import time
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from lib.controllers.purepursuit import PurePursuit
from lib.controllers.pid import PID
from lib.drive import Driver


class Controller:
    def __init__(self):
        self.pid = PID(2.9, 2.3, 0.0)
        self.desire = 'straight'
        self.mode = 'manual'
        self.speedlimit = 30
        self.speed = 0
        self.overspeed = False
        self.purepursuit = PurePursuit()
        self.driver = Driver()
        self.acc = 0.0
        self.steer = 0.0
        self.steer_last = None
        self.trajectory = None
        self.speed_plan = None
        self.acc_limit = [-0.3, 0.3]
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
        condition_dict_sub = SharedMemoryDict(name='condition', size=1024)
        plan_dict_sub = SharedMemoryDict(name='plan', size=1024)
        
        if 'speed' in condition_dict_sub.keys():
            if condition_dict_sub['speedlimit'] is not None:
                self.speedlimit = condition_dict_sub['speedlimit']
            self.speed = condition_dict_sub['speed']
            self.overspeed = condition_dict_sub['overspeed']
        if 'desire' in option_dict_sub.keys():
            self.desire = option_dict_sub['desire']
            self.mode = option_dict_sub['mode']
        if 'planv' in plan_dict_sub.keys():
            self.speed_plan = plan_dict_sub['planv']
        if 'trajectory' in plan_dict_sub.keys():
            self.trajectory = plan_dict_sub['trajectory']
            
    def run(self):
        if self.trajectory is not None:
            self.steer = self.purepursuit.run(self.trajectory, self.speed)
        if self.speed_plan is not None:
            speed_target = min(self.speed_plan, self.speedlimit)
        else:
            speed_target = self.speedlimit
        self.pid.update_e(speed_target - self.speed)
        self.acc = self.pid.get_a()
        self.acc = self.limit_accelerate(self.acc)
        if self.overspeed == True:
            self.acc = 0.1
        if self.desire in ['changelaneleft', 'changelaneright']:
            self.acc = 0
        if self.mode == 'manual':
            self.driver.drive(0.5, 0.5)
        elif self.mode == 'latcontrol' and self.steer == self.steer:
            self.driver.drive(self.steer + 0.5, 0.5)
        elif self.mode == 'longcontrol' and self.acc == self.acc:
            self.driver.drive(0.5, self.acc + 0.5)
        elif (self.mode == 'AP' or self.mode == 'NAV') and self.steer == self.steer and self.acc == self.acc:
            self.driver.drive(self.steer + 0.5, self.acc + 0.5)
        else:
            self.driver.drive(0.5, 0.5)
        
    def limit_accelerate(self, acc):
        if acc > self.acc_limit[1]:
            acc = self.acc_limit[1]
        elif acc < self.acc_limit[0]:
            acc = self.acc_limit[0]
        return acc
    
    def exit(self):
        self.driver.end()
            