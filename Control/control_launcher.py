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
from Planning.Cruise import Cruise
from Control.controllers.PID_controller import PID
from Condition.truck_condition import Truck
from Planning.bazier_optimizer import point_on_bezier_curve
from Perception.LaneDetection.transform import trans_translate, trans_rotate
from Common.iodata import load_pkl, save_pkl

driver = Driver()
truck = Truck()
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
    speed_dict = load_pkl(os.path.join(project_path, 'temp/speed.pkl'))

    if lane_dict is not None:
        line_m = lane_dict['line_m']
        lane_width = lane_dict['lane_width']
    if speed_dict is not None:
        if len(speed_dict):
            truck.speed = speed_dict['speed']
    if acc is not None and ang is not None:
        truck.update(ang * np.pi * 2, acc, 0.024)
    truck.publish()

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
            trajectory_change = trans_translate(trajectory_change, -truck.dx, -truck.dy)
            trajectory_change = trans_rotate(trajectory_change, -truck.dtheta)
            trajectory_change = trajectory_change[np.where(trajectory_change[:, 0] >= 0)[0], :]
            if len(trajectory_change) == 0:
                trajectory_change = None
            trajectory = trajectory_change

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break

    np.save(os.path.join(project_path, 'temp/trajectory.npy'), trajectory)

    if trajectory is not None:
        if condition_dict is not None:
            acc, ang = Cruise(vertical_pid, horizontal_pid, condition_dict, trajectory)
        if speed_dict is not None and len(speed_dict) == 2:
            if speed_dict['over_speed'] == 1:
                acc = 0.1
        if trajectory_change is not None:
            acc = 0

        control_dict = {'acc': acc, 'ang': ang}
        save_pkl(os.path.join(project_path, 'temp/control.pkl'), control_dict)

        time.sleep(0.01)
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
            if option_dict['power'] == 0:
                driver.end()
                time.sleep(0.1)
                break
