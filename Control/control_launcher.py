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
from Planning.bazier_optimizer import point_on_bezier_curve
from Perception.LaneDetection.transform import trans_translate, trans_rotate
from Condition.truck_condition import Truck

driver = Driver()
truck = Truck()
horizontal_pid = PID(0.3, 0.001, 0.001)  # 初始化横向PID控制算法
vertical_pid = PID(0.3, 0.0, 0.05)  # 初始化纵向PID控制算法
trajectory = None
option_list = None
condition_list = None
speed_list = None
acc, ang = None, None
trajectory_change = None
lane_width = 3.6

while True:
    try:
        trajectory = np.load(os.path.join(project_path, 'temp/trajectory.npy'))
    except ValueError:
        print('lost trajectory data')

    try:
        condition_file = open(os.path.join(project_path, 'temp/condition.pkl'), 'rb')
        condition_list = pickle.load(condition_file)
        condition_file.close()
    except EOFError:
        print('lost condition data')

    try:
        option_list = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
    except ValueError:
        print('lost option data')

    try:
        speed_list = np.loadtxt(os.path.join(project_path, "temp/speed.txt"), dtype=bytes).astype(float)
    except ValueError:
        print('lost speed data')

    if len(speed_list):
        speed, over_speed = speed_list[0], speed_list[0]
        truck.speed = speed
    if acc is not None and ang is not None:
        truck.update(ang * np.pi * 2, acc, 0.024)
    if trajectory is not None and option_list is not None and len(option_list):
        if option_list[1] in [3, 4] and trajectory_change is None:  # 辅助变道规划
            defaut_change_distance = 25
            if option_list[1] == 3:
                line_target = trajectory - [[0, lane_width]]
            else:
                line_target = trajectory + [[0, lane_width]]

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

    canva = np.zeros([300, 120, 3])
    if trajectory is not None:
        for i in range(len(trajectory)):
            cv2.circle(canva,
                       (int(trajectory[i][1] * 6 + 120 // 2), int(300 - trajectory[i][0] * 6)),
                       radius=1,
                       thickness=-1,
                       color=(100, 100, 100))

    if trajectory is not None:
        if condition_list is not None:
            acc, ang = Cruise(vertical_pid, horizontal_pid, condition_list, trajectory)
        if speed_list is not None and len(speed_list) == 2:
            if speed_list[1] == 1:
                acc = 0
        if trajectory_change is not None:
            acc = 0
        control_list = {'acc': acc, 'ang': ang}
        control_file = open(os.path.join(project_path, 'temp/control.pkl'), 'wb')
        pickle.dump(control_list, control_file)
        control_file.close()

        time.sleep(0.01)
        if option_list is not None and len(option_list):
            if option_list[0] == 0:
                driver.drive(0.5, 0.5)
            elif option_list[0] == 1 and ang == ang:
                driver.drive(ang + 0.5, 0.5)
            elif option_list[0] == 2 and acc == acc:
                driver.drive(0.5, acc + 0.5)
            elif option_list[0] == 3 and ang == ang and acc == acc:
                driver.drive(ang + 0.5, acc + 0.5)
            else:
                driver.drive(0.5, 0.5)
            if option_list[2] == 0:
                driver.end()
                time.sleep(0.1)
                break
