'''
自动驾驶主程序
'''

title_bar_height = 47

import warnings
warnings.filterwarnings('ignore')
import os, sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

perception_path = os.path.abspath(os.path.join(project_path, 'Perception'))
sys.path.insert(0, perception_path)
from ObjectDetection.yolov6_trt import YOLOPredictor
from ObjectDetection.cipv_notice import cipv_notice
from ObjectDetection.ByteTrack.tracker.byte_tracker import BYTETracker
#from LaneDetection.clrnet_trt import CLRNet
from LaneDetection.FCW import FCW
from StopLineDetection.Linedetection import line_filter
#from DepthEstimation.monodepth2.monodepth import load_monodepth, depth_detect
'''
sys.path.insert(0,'D:/autodrive/PaddleSeg')
from deploy.python.seg_infer import SegPredictor
'''
planning_path = os.path.abspath(os.path.join(project_path, 'Planning'))
sys.path.insert(0, planning_path)
from FSMPlaning import FSMPlanner, PlanTrigger
from Cruise import Cruise
from Follow import Follow

control_path = os.path.abspath(os.path.join(project_path, 'Control'))
sys.path.insert(0, control_path)
from drive import driver, end, Truck, Info
from controllers.PID_controller import PID
from controllers.fuzzy_controller import fuzzy_initialization

navigation_path = os.path.abspath(os.path.join(project_path, 'Navigation'))
sys.path.insert(0, navigation_path)
from BevDraw import draw_bev, print_info
from Navigation_Process import nav_process

from ocr_tool import speed_detect
from grab_screen import grab_screen
import cv2
import time
import numpy as np
import torch
from paddleocr import PaddleOCR
from ABS import TTC
from camera import cam, backcam_left, backcam_right
from threading import Thread
import threading
import win32api
last_time = time.time()

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda:0'

CAM = cam()  # 定义相机
CAM_BL = backcam_left()  # 定义左后视镜
CAM_BR = backcam_right()  # 定义右后视镜
#encoder, depth_decoder, feed_height, feed_width = load_monodepth(device) # 加载深度估计模型
yolo_engine_path = os.path.abspath(os.path.join(project_path, 'Engines', 'yolov6s_bdd_60.engine'))
yolopredictor = YOLOPredictor(engine_path = yolo_engine_path)
#llamas_engine_path = os.path.abspath(os.path.join(project_path, 'Engines', 'llamas_dla34.engine'))
#clrnet = CLRNet(llamas_engine_path)
ocr = PaddleOCR(enable_mkldnn=True, use_tensorrt=True, use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
#segpredictor = SegPredictor("D:/autodrive/PaddleSeg/output/inference_model_mobile_seg/deploy.yaml")
fsmplanner = FSMPlanner("autotruck")
planetrigger = PlanTrigger()

track_thresh = 0.4 # 跟踪阈值
track_buffer = 15 # 跟踪缓冲区
match_thresh = 0.75 # 匹配阈值
frame_rate = 5 # 帧率
aspect_ratio_thresh = 1.6 # 长宽比阈值
min_box_area = 10 # 最小框面积
mot20_check = False # 是否使用mot20检测
vehicle_tracker = BYTETracker(track_thresh, track_buffer, match_thresh, mot20_check, frame_rate)

truck = Truck()  # 初始化汽车状态模型
info = Info()  # 初始化信息储存
tracks = [[1]]

horizontal_pid = PID(0.02, 0.0005, 0.01)  # 初始化横向PID控制算法
vertical_pid = PID(0.05, 0, 0.1)  # 初始化纵向PID控制算法
vertical_fuzzy = fuzzy_initialization()  # 初始化模糊控制跟踪算法

pts1 = np.array([[0, 0], [900, 0],
                 [0, 280], [900, 280]], dtype=np.float32)
pts2 = np.array([[0, 0], [900, 0],
                 [350, 900], [550, 900]], dtype=np.float32)
M = cv2.getPerspectiveTransform(pts1, pts2)  # 原图的透视变换矩阵

# 初始化参数
speed = 0
truck.ang = 0.5
speed_limit = 0
stop = 0
car_stop = 0
red_stop = 0
state = 'Cruise'

class MyThread(threading.Thread):
    def __init__(self, func, args=()):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args
    def run(self):
        self.result = self.func(*self.args)
    def get_result(self):
        threading.Thread.join(self)  # 等待线程执行完毕
        try:
            return self.result
        except Exception:
            return None

cipv = None
img = cv2.cvtColor(grab_screen(region=(0, title_bar_height, 1359, 767 + title_bar_height)), cv2.COLOR_RGB2BGR)
im0 = img.copy()
while True:
    if 'last_time' in locals().keys():
        refer_time = time.time() - last_time
    else:
        refer_time = 0.18
    last_time = time.time()

    img = cv2.cvtColor(grab_screen(region=(0, title_bar_height, 1359, 767 + title_bar_height)), cv2.COLOR_RGB2BGR)
    im0 = img.copy()

    # 环境感知
    if np.average(img) <= 50: weather = 'dark'  # 判断天气
    elif np.average(img) >= 200: weather = 'snowy'
    else: weather = 'white'

    '''
    img_seg = segpredictor.run(cv2.resize(img[60:570, 185:1095, :], (1280, 720)))
    img_seg = cv2.cvtColor(np.asarray(img_seg), cv2.COLOR_RGB2BGR)
    cv2.imshow('test', cv2.resize(img_seg, (416, 234)))
    '''

    # 目标检测
    im1 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (1280, 720)).copy()  # 检测结果画布
    im1, obstacles, objs, tracks, traffic_light = yolopredictor.inference(cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (1280, 720)), CAM, CAM_BL, CAM_BR, im1, vehicle_tracker, tracks, refer_time, conf=0.4)

    # 车道线检测(暂时取消)
    bev_lanes = []
    '''
    if traffic_light == None:
        im1, bev_lanes = clrnet.forward(cv2.resize(img, (1280, 720)), im1, CAM)  # 传入RGB图像
    else:
        bev_lanes = []
    '''

    # 变道处理
    if info.change_lane / 15 != info.change_lane_dest:
        if info.change_lane / 15 < info.change_lane_dest:
            info.change_lane = info.change_lane + 1
            info.direction = 1
        else:
            info.change_lane = info.change_lane - 1
            info.direction = -1
    else:
        info.direction = 0

    # 导航图处理
    navmap = cv2.cvtColor(img[610:740, 580:780, :], cv2.COLOR_RGB2BGR)  # 截取导航地图[130, 200, 3]
    navmap[:, 0:50, :] = np.zeros([130, 50, 3])
    navmap[:, 150:200, :] = np.zeros([130, 50, 3])
    nav_line, curve_speed_limit = nav_process(navmap, truck, info, cipv)
    im1 = np.uint8(im1)

    # 自车状态监控
    bar = cv2.cvtColor(img[750:768, 545:595, :], cv2.COLOR_RGB2BGR)  # 截取速度条[18, 50, 3]
    speed = speed_detect(ocr, bar, truck.speed)
    speed_limit = curve_speed_limit

    # 车辆前向物体检测线(暂时替代车道线)
    bev_lanes.extend(FCW(nav_line, speed))

    # 障碍物及停止线检测
    obstacles, cipv = cipv_notice(obstacles, bev_lanes)
    if traffic_light is not None:
        if traffic_light.tlcolor == 1 or traffic_light.tlcolor == 2:  # 检测到红灯或黄灯时再检测停止线
            stop_line = line_filter(cv2.resize(img, (1280, 720)), M)
        else:
            stop_line = None
    else:
        stop_line = None
    '''
    if weather == 'white' and truck.speed <= 45:
        depth_img = depth_detect(encoder, depth_decoder, img, feed_height, feed_width)
    '''

    # 决策
    if obstacles is not None:
        state_trigger = planetrigger.update_trigger(fsmplanner.state, obstacles, bev_lanes, info, truck)
        if state_trigger is not None:
            fsmplanner.trigger(state_trigger)
            state = fsmplanner.state

    # 控制
    acc = 0.5
    ang = 0.5

    acc, ang = Cruise(vertical_pid, horizontal_pid, truck, speed_limit, nav_line, info)
    if (state == 'Follow' or state == 'Passing') and cipv is not None:
        acc = Follow(cipv, vertical_fuzzy, truck)
    if state == 'Passing':
        if planetrigger.change_lane_state == 1:
            info.change_lane_dest = -1
            if info.change_lane / 15 == info.change_lane_dest:
                planetrigger.change_lane_state = 2
        if planetrigger.change_lane_state == 2:
            info.update(2)
        if planetrigger.change_lane_state == 3:
            if time.time() - planetrigger.t >= 4:
                planetrigger.change_lane_state = 0
                info.change_lane_dest = 0
                info.update(1)
                planetrigger.t_sheld = 0

    if info.activeAP:
        driver(ang, acc)
    elif info.AP_exit_reason == 1:
        driver(0.5, 0.75)
    else:
        driver(0.5, 0.5)

    truck.update(ang, acc, refer_time, speed)

    # BEV绘制
    bevmap = draw_bev(nav_line, obstacles, traffic_light, bev_lanes, stop_line)
    bevmap = bevmap[160:, 160:640, :]
    bevmap = cv2.resize(bevmap, (416, 346))
    bevmap = print_info(bevmap, refer_time, truck, speed_limit, state, weather, info, planetrigger)

    img_show = cv2.vconcat([cv2.resize(im1, (416, 234)), bevmap])
    '''
    if weather == 'dark' or truck.speed >= 45:
        img_show = cv2.vconcat([cv2.resize(im1, (416, 234)),
                                cv2.resize(bevmap, (416, 346))])
    else:
        img_show = cv2.vconcat([cv2.resize(im1, (416, 234)),
                                cv2.cvtColor(cv2.resize(depth_img, (416, 234)), cv2.COLOR_GRAY2BGR),
                                bevmap])
    '''
    cv2.imshow('detect', img_show)

    # ctrl+Q退出
    if cv2.waitKey(25) & 0xFF == ord('q') or (win32api.GetAsyncKeyState(0x51) and win32api.GetAsyncKeyState(0x11)):
        cv2.destroyAllWindows()
        driver(0.5, 0.5)
        break

    # 1键退出自动驾驶
    if win32api.GetAsyncKeyState(0x31):
        info.activeAP = False
        info.AP_exit_reason = 0
        planetrigger.t_sheld = 0
    # 6键激活自动驾驶及切换道路类型
    if win32api.GetAsyncKeyState(0x36):
        planetrigger.t_sheld = 0
        if not info.activeAP:
            info.activeAP = True
            info.change_lane_dest = 0
            info.update(0)
        else:
            if info.roads_type == 0:
                info.update(1)
            else:
                info.update(0)

time.sleep(0.1)
end()
