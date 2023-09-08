import os
import numpy as np
import sys
import time

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Message.iodata import load_pkl, save_pkl
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
        condition_dict = load_pkl(os.path.join(project_path, 'Message/temp/condition.pkl'))
        option_dict = load_pkl(os.path.join(project_path, 'Message/temp/option.pkl'))
        speed_plan = load_pkl(os.path.join(project_path, 'Message/temp/planv.pkl'))
        if condition_dict is not None:
            if condition_dict['speedlimit'] is not None:
                self.speedlimit = condition_dict['speedlimit']
            self.speed = condition_dict['speed']
            self.overspeed = condition_dict['overspeed']
        if option_dict is not None:
            self.desire = option_dict['desire']
            self.mode = option_dict['mode']
        if speed_plan is not None:
            self.speed_plan = speed_plan
        try:
            self.trajectory = np.load(os.path.join(project_path, 'Message/temp/trajectory.npy'), allow_pickle=False)
        except ValueError:
            print('lost trajectory data')
            
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
        elif self.mode == 'AP' and self.steer == self.steer and self.acc == self.acc:
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
            