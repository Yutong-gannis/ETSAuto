import warnings
import os
import sys
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
project_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_path)
from Perception.LaneDetection.onnx_infer import Bev_Lanedet
from Perception.LaneDetection.planregister import PlanRegister
from Screen.grab_screen import ScreenGraber
from Navigation.Navigation_Process import nav_process
from Common.iodata import load_pkl, save_pkl

nav_line = None
lane_path = os.path.abspath(os.path.join(project_path, 'weights/bevlanedet/resnet18_0.5/ep030.onnx'))
lanedet = Bev_Lanedet(lane_path)
plan_register = PlanRegister()
screengraber = ScreenGraber()
fps = 0.06
fps_list = []
im0 = screengraber.update()

i = 0
while True:
    t0 = time.time()
    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))

    pool = ThreadPoolExecutor(max_workers=3)
    thread1 = pool.submit(lanedet.infer, im0)
    thread2 = pool.submit(screengraber.update)
    # thread3 = pool.submit(nav_process, cv2.cvtColor(im0[610:740, 580:780, :], cv2.COLOR_RGB2BGR))
    im0 = thread2.result()
    # nav_line = thread3.result()
    line_l, line_r, line_m, lane_width = thread1.result()
    pool.shutdown()

    if line_m is not None:
        line_m = plan_register.update(line_m)
    # elif nav_line[0, 1] <= 0.1:
    #     line_m = plan_register.update(nav_line)

    lane_dict = {'line_l': line_l, 'line_r': line_r, 'line_m': line_m, 'lane_width': lane_width}
    save_pkl(os.path.join(project_path, 'temp/line.pkl'), lane_dict)

    '''
    bevmap = np.zeros((300, 120))
    if len(nav_line):
        for nav_pt in nav_line:
            cv2.circle(bevmap, (60 + int(nav_pt[1]*6), 300-int(nav_pt[0]*6)), radius=1, color=(100, 100, 100),
                       thickness=-1)
    cv2.imshow('bevmap', bevmap)
    '''
    t1 = time.time()
    fps = t1 - t0
    if len(fps_list) == 10:
        fps_list.pop(0)
    fps_list.append(fps*1000)
    print('fps', fps)

    if cv2.waitKey(25) & 0xFF == ord('q') or (option_dict is not None and option_dict['power'] == 0):
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break
