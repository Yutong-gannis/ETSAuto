import warnings
import os
import sys
import time
import cv2
import pickle
import numpy as np
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)
from Perception.LaneDetection.onnx_infer import Bev_Lanedet
from Perception.LaneDetection.planregister import PlanRegister
from Screen.grab_screen import ScreenGraber
from Common.initialization import init
from Navigation.Navigation_Process import nav_process

truck, tracks, M, speed_limit, state, nav_line = init()
lane_path = os.path.abspath(os.path.join(project_path, 'weights/bevlanedet/ep020.onnx'))
lanedet = Bev_Lanedet(lane_path)
plan_register = PlanRegister()
t0 = time.time()
screengraber = ScreenGraber()
line_l, line_r, line_m, trajectory = None, None, None, None
trajectory_change = None
speed_list = [0, 0]
fps = 0.06
fps_list = []
option_list = [0, 0, 1]
control_list = {'acc': 0.5, 'ang': 0.5}

im0 = screengraber.update()

i = 0
while True:
    t0 = time.time()

    try:
        option_list_temp = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
        if len(option_list_temp):
            option_list = option_list_temp
    except ValueError:
        print('lost option data')

    try:
        control_file = open(os.path.join(project_path, 'temp/condition.pkl'), 'rb')
        control_list = pickle.load(control_file)
        control_file.close()
    except EOFError:
        print('lost control data')

    pool = ThreadPoolExecutor(max_workers=3)
    thread1 = pool.submit(lanedet.infer, im0)
    thread2 = pool.submit(screengraber.update)
    thread3 = pool.submit(nav_process, cv2.cvtColor(im0[610:740, 580:780, :], cv2.COLOR_RGB2BGR))
    im0 = thread2.result()
    nav_line = thread3.result()
    line_l, line_r, line_m, lane_width = thread1.result()
    pool.shutdown()

    screengraber.publish(project_path)

    try:
        speed_list = np.loadtxt(os.path.join(project_path, "temp/speed.txt"), dtype=bytes).astype(float)
    except ValueError:
        print('lost speed data')
    if len(speed_list):
        speed, over_speed = speed_list[0], speed_list[0]
        truck.speed = speed
    if control_list is not None:
        truck.update(control_list['acc'], control_list['ang'], fps)
    truck.publish()

    if line_m is not None:
        trajectory = plan_register.update(line_m)
    elif nav_line[0, 1] <= 0.1:
        trajectory = plan_register.update(nav_line)

    np.save('temp/trajectory.npy', trajectory)

    if line_l is not None:
        np.save('temp/line_l.npy', line_l)
        np.save('temp/line_r.npy', line_r)

    t1 = time.time()
    fps = t1 - t0
    if len(fps_list) == 10:
        fps_list.pop(0)
    fps_list.append(fps*1000)

    print(fps)
    line_l, line_r, line_m, trajectory = None, None, None, None
    # ctrl+Q退出
    if cv2.waitKey(25) & 0xFF == ord('q') or option_list[2] == 0:
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break
