import sys, os
import pycuda.autoinit
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

sys.path.insert(0, os.path.abspath(os.path.join(project_path, 'Perception')))
from ObjectDetection.yolov6_trt import YOLOPredictor
from ObjectDetection.ByteTrack.tracker.byte_tracker import BYTETracker
from LaneDetection.clrnet_trt import CLRNet

sys.path.insert(0, os.path.abspath(os.path.join(project_path, 'Control')))
from drive import Truck, Info
from controllers.PID_controller import PID
from controllers.fuzzy_controller import fuzzy_initialization

sys.path.insert(0, os.path.abspath(os.path.join(project_path, 'Perception')))
from FSMPlaning import FSMPlanner, PlanTrigger

import numpy as np
import cv2
import yaml
from camera import cam, backcam_left, backcam_right
from paddleocr import PaddleOCR
import torch


def Perception_init(project_path):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    device = 'cuda:0'

    ocr = PaddleOCR(enable_mkldnn=True, use_tensorrt=True, use_angle_cls=False, lang="en", use_gpu=False,
                    show_log=False)
    with open(os.path.abspath(os.path.join(project_path, 'Perception/ObjectDetection/data/obj.yaml')),
              encoding='utf-8') as f:
        obj_cfg = yaml.load(f, Loader=yaml.FullLoader)
    vehicle_tracker = BYTETracker(obj_cfg['track'])

    return obj_cfg, vehicle_tracker, ocr


def Planner_init():
    fsmplanner = FSMPlanner("autotruck")
    #with open(os.path.abspath(os.path.join(project_path, 'Planning/scenario.yaml')), encoding='utf-8') as f:
    #    scenario_register = yaml.load(f, Loader=yaml.FullLoader)
    return fsmplanner


def init():
    CAM = cam()  # 定义相机
    CAM_BL = backcam_left()  # 定义左后视镜
    CAM_BR = backcam_right()  # 定义右后视镜
    truck = Truck()  # 初始化汽车状态模型
    tracks = [[1]]
    intersection_condition = 0
    horizontal_pid = PID(0.02, 0.0005, 0.01)  # 初始化横向PID控制算法
    vertical_pid = PID(0.03, 0, 0.1)  # 初始化纵向PID控制算法
    vertical_fuzzy = fuzzy_initialization()  # 初始化模糊控制跟踪算法
    planetrigger = PlanTrigger()
    pts1 = np.array([[0, 0], [900, 0],
                     [0, 280], [900, 280]], dtype=np.float32)
    pts2 = np.array([[0, 0], [900, 0],
                     [350, 900], [550, 900]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts1, pts2)  # 原图的透视变换矩阵
    info = Info()  # 初始化信息储存
    speed_limit = 0
    state = 'Cruise'
    nav_line = None
    return info, CAM, CAM_BL, CAM_BR, truck, tracks, horizontal_pid, vertical_pid, vertical_fuzzy, M, speed_limit, state, nav_line, intersection_condition, planetrigger
