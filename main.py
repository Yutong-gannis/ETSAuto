import warnings
import os
import sys
import time
import cv2
import numpy as np
import importlib
from concurrent.futures import ThreadPoolExecutor

ENGINE = 'onnx'
VERSION = 2

warnings.filterwarnings('ignore')
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)

Postprocess_dir = importlib.import_module('ModelPlan.Postprocess.Postprocess_' + str(VERSION))

from ModelPlan.load_model import load_model
from ModelPlan.planregister import PlanRegister
from Camera.transform import Calibration
from Planning.Cruise import Cruise
from User.interface import UserInterface, DevInterface
from Screen.grab_screen import ScreenGraber
from Common.initialization import init

truck, tracks, horizontal_pid, vertical_pid, M, speed_limit, state, nav_line = init()
supercombo_path = os.path.abspath(os.path.join(project_path, 'weights', 'supercombo_'+str(VERSION) + '.' + ENGINE))
supercombo = load_model(supercombo_path, VERSION, ENGINE)
postproccess = Postprocess_dir.ModelOutput()
user_interface = UserInterface()
develop_interface = DevInterface()
last_time = time.time()
plan_register = PlanRegister()

rpy_calib_pred = np.array([0.00018335809, 0.034165092, -0.014245722]) / 2
calibration = Calibration(rpy_calib_pred, plot_img_width=1360, plot_img_height=768)
screengraber = ScreenGraber()
model_output = None
plan_position = None
plan_yaw = None
speed_list = [0, 0]
fps_list = []
option_list = [0, 0, 1]

im0 = screengraber.update()
nav = cv2.cvtColor(im0[610:740, 580:780, :], cv2.COLOR_RGB2BGR)

i = 0
while True:
    if 'last_time' in locals().keys():
        refer_time = time.time() - last_time
    else:
        refer_time = 0.18
    last_time = time.time()

    try:
        option_list_temp = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
        if len(option_list_temp):
            option_list = option_list_temp
    except ValueError:
        print('lost option data')

    model_output = supercombo.infer(im0, option_list[1])

    pool = ThreadPoolExecutor(max_workers=3)
    thread2 = pool.submit(postproccess.process, model_output)
    thread3 = pool.submit(screengraber.update)

    im0 = thread3.result()
    plan_position, plan_velocity, plan_acc, plan_yaw, plan_yaw_rate, plan_angle, lanes, edges, leaders, pose = thread2.result()

    plan_position = plan_register.update(plan_position)
    pool.shutdown()

    screengraber.publish(project_path)
    nav = cv2.cvtColor(im0[610:740, 580:780, :], cv2.COLOR_RGB2BGR)

    try:
        speed_list = np.loadtxt(os.path.join(project_path, "temp/speed.txt"), dtype=bytes).astype(float)
    except ValueError:
        print('lost speed data')
    if len(speed_list):
        speed, over_speed = speed_list[0], speed_list[0]
        truck.speed = speed
    acc, ang = 0.5, 0.5

    if plan_position is not None and len(plan_position):
        acc_1, ang_1 = Cruise(vertical_pid, horizontal_pid, truck, plan_velocity, plan_position)
        control_list = np.concatenate((plan_acc[0:10], plan_yaw_rate[0:10, 2]), axis=0)
        control_list = np.append(control_list, acc_1)
        control_list = np.append(control_list, ang_1)
        control_list = np.append(control_list, time.time())
        np.save('temp/control.npy', control_list)

    t2 = time.time()
    fps = (t2 - last_time) * 1000
    if len(fps_list) == 10:
        fps_list.pop(0)
    fps_list.append(fps)
    user_interface.show(im0, lanes, plan_position, edges, leaders, option_list, truck, round(sum(fps_list)/10, 1))
    # develop_interface.show(lanes, edges, plan_position)

    # ctrl+Q退出
    if cv2.waitKey(25) & 0xFF == ord('q') or option_list[2] == 0:
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break
