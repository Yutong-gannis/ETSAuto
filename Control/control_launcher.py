import time
import os
import sys
import math
import scipy
import sympy as sp
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from drive import Driver
from Control.Cruise import Cruise
from Control.plans.changelane import ChangeLane_Helper
from Control.controllers.PID_controller import PID
from Control.planregister import PlanRegister
from Control.transform import trans_translate, trans_rotate
from Common.iodata import load_pkl, save_pkl

driver = Driver()
horizontal_pid = PID(0.3, 0.001, 0.001)  # 初始化横向PID控制算法
vertical_pid = PID(0.3, 0.0, 0.05)  # 初始化纵向PID控制算法
plan_register = PlanRegister()
changelane_helper = ChangeLane_Helper()
line_m = None
trajectory = None
acc, ang = None, None
ang_last = None
lane_width = 3.6

while True:
    lane_dict = load_pkl(os.path.join(project_path, 'temp/line.pkl'))
    condition_dict = load_pkl(os.path.join(project_path, 'temp/condition.pkl'))
    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))
    
    if lane_dict is not None:
        line_m = lane_dict['line_m']
        line_l = lane_dict['line_l']
        lane_width = lane_dict['lane_width']
    
    if line_m is not None:
        trajectory = line_m
    if trajectory is not None and option_dict is not None and len(option_dict):
        trajectory_change, plan_register = changelane_helper.update(trajectory, line_l, plan_register, lane_width, option_dict, condition_dict)
        if trajectory_change is not None:
            trajectory = trajectory_change

    np.save(os.path.join(project_path, 'temp/trajectory.npy'), trajectory)
     
    if trajectory is not None and len(trajectory) >= 50:
        trajectory = plan_register.update(trajectory[:50, :])
        if condition_dict is not None:
            acc, ang = Cruise(vertical_pid, horizontal_pid, condition_dict, trajectory)
            if condition_dict['overspeed'] == True:
                acc = 0.1
        if trajectory_change is not None:
            acc = 0
        
        if option_dict is not None and len(option_dict):
            if option_dict['mode'] == 0:
                driver.drive(0.5, 0.5)
            elif option_dict['mode'] == 1 and ang == ang:
                driver.drive(ang + 0.5, 0.5)
            elif option_dict['mode'] == 2 and acc == acc:
                driver.drive(0.5, acc + 0.5)
            elif option_dict['mode'] == 3 and ang == ang and acc == acc:
                driver.drive(ang + 0.5, acc + 0.5)
            else:
                driver.drive(0.5, 0.5)

    time.sleep(0.05)
    ang_last = ang

    if option_dict is not None and len(option_dict):
        if option_dict['power'] == 0:
            driver.end()
            time.sleep(0.1)
            break
