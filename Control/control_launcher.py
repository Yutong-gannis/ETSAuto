import time
import os
import sys
import numpy as np
import pickle
import cv2

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from drive import Driver
from Control.Cruise import Cruise
from Control.controllers.PID_controller import PID
from Control.optimizers.bazier_optimizer import point_on_bezier_curve
from Perception.LaneDetection.transform import trans_translate, trans_rotate
from Common.iodata import load_pkl, save_pkl

driver = Driver()
horizontal_pid = PID(0.3, 0.001, 0.001)  # 初始化横向PID控制算法
vertical_pid = PID(0.3, 0.0, 0.05)  # 初始化纵向PID控制算法
line_m = None
trajectory = None
acc, ang = None, None
trajectory_change = None
lane_width = 3.6

while True:
    lane_dict = load_pkl(os.path.join(project_path, 'temp/line.pkl'))
    condition_dict = load_pkl(os.path.join(project_path, 'temp/condition.pkl'))
    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))

    if lane_dict is not None:
        line_m = lane_dict['line_m']
        lane_width = lane_dict['lane_width']

    trajectory = line_m
    if trajectory is not None and option_dict is not None and len(option_dict):
        if option_dict['desire'] in [3, 4] and trajectory_change is None:  # 辅助变道规划
            defaut_change_distance = 25
            trajectory_theta = np.arctan((trajectory[3, 1] - trajectory[0, 1]) / (trajectory[3, 0] - trajectory[0, 0]))
            if option_dict['desire'] == 3:
                line_target = trajectory - [[-lane_width * np.sin(trajectory_theta), lane_width * np.cos(trajectory_theta)]]
            else:
                line_target = trajectory + [[-lane_width * np.sin(trajectory_theta), lane_width * np.cos(trajectory_theta)]]

            P0 = [0, 0]
            P1 = trajectory[5, :]
            P2 = line_target[5 + defaut_change_distance * 2, :]
            P3 = line_target[5 + defaut_change_distance * 2 + 5, :]

            a = 1 / 4
            Ay = P1[1]
            Ax = P1[0] + a * (P2[0] - P0[0])
            A = np.array([Ax, Ay])

            b = 1 / 4
            By = P2[1]
            Bx = P2[0] - b * (P3[0] - P1[0])
            B = np.array([Bx, By])

            ts = np.linspace(0, 1, defaut_change_distance * 2 + 1)

            Q = np.zeros((defaut_change_distance * 2 + 1, 2))
            for i, t in enumerate(ts):
                Q[i, :] = point_on_bezier_curve([P1, A, B, P2], t)
            trajectory_change = np.concatenate((trajectory[:5, :], Q), axis=0)

        elif trajectory_change is not None:
            print(len(trajectory_change))
            if condition_dict is not None:
                print(condition_dict['dtheta'])
                trajectory_change = trans_rotate(trajectory_change, -condition_dict['dtheta'])
                trajectory_change = trans_translate(trajectory_change, -condition_dict['dx'], -condition_dict['dy'])
            trajectory_change = trajectory_change[np.where(trajectory_change[:, 0] >= 0)[0], :]
            if len(trajectory_change) <= 20:
                trajectory_change = None
            trajectory = trajectory_change
    
    np.save(os.path.join(project_path, 'temp/trajectory.npy'), trajectory)

    if trajectory is not None:
        if condition_dict is not None:
            acc, ang = Cruise(vertical_pid, horizontal_pid, condition_dict, trajectory)
            print(acc)
            print(ang)
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
    if option_dict is not None and len(option_dict):
        if option_dict['power'] == 0:
            driver.end()
            time.sleep(0.1)
            break
