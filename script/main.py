import warnings
warnings.filterwarnings('ignore')
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

perception_path = os.path.abspath(os.path.join(project_path, 'Perception'))
sys.path.insert(0, perception_path)
from ObjectDetection.cipv_notice import cipv_notice
from ObjectDetection.yolov6_trt import YOLOPredictor
from LaneDetection.clrnet_trt import CLRNet
from LaneDetection.FCW import FCW
from SpeedOCR.ocr_tool import speed_detect
from StopLineDetection.Linedetection import line_filter
from SceneClassifier.infer import weather_infer

planning_path = os.path.abspath(os.path.join(project_path, 'Planning'))
sys.path.insert(0, planning_path)
from Cruise import Cruise
from Follow import Follow

control_path = os.path.abspath(os.path.join(project_path, 'Control'))
sys.path.insert(0, control_path)
from drive import driver, end

navigation_path = os.path.abspath(os.path.join(project_path, 'Navigation'))
sys.path.insert(0, navigation_path)
from BevDraw import draw_bev, print_info
from Navigation_Process import nav_process

from initialization import init, Perception_init, Planner_init
from grab_screen import grab_screen
import cv2
import time
import numpy as np
from parallel import MyThread
from concurrent.futures import ThreadPoolExecutor
import win32api

device = 'cuda:0'
obj_cfg, vehicle_tracker, ocr, weather_classifier = Perception_init(project_path)
fsmplanner = Planner_init()
info, CAM, CAM_BL, CAM_BR, truck, tracks, horizontal_pid, vertical_pid, vertical_fuzzy, M, speed_limit, state, nav_line, intersection_condition, planetrigger = init()
yolo_engine_path = os.path.abspath(os.path.join(project_path, 'Engines', 'yolov6s_bdd_300.engine'))
clrnet_engine_path = os.path.abspath(os.path.join(project_path, 'Engines', 'llamas_dla34.engine'))
yolopredictor = YOLOPredictor(engine_path=yolo_engine_path)
clrnet = CLRNet(clrnet_engine_path)
last_time = time.time()
weather_time = time.time()
img, im0 = grab_screen()

weather = 'clear'
scene = 'highway'
cipv = None
while True:
    if 'last_time' in locals().keys():
        refer_time = time.time() - last_time
    else:
        refer_time = 0.18
    last_time = time.time()
    # 环境感知
    if np.average(img) <= 50:
        timeofday = 'dark'  # 判断天气
    elif np.average(img) >= 200:
        timeofday = 'snowy'
    else:
        timeofday = 'white'

    # 目标检测
    im1 = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (1280, 720)).copy()  # 检测结果画布
    im1, obstacles, objs, tracks, traffic_light = yolopredictor.inference(
        cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (1280, 720)), CAM, CAM_BL, CAM_BR, im1, vehicle_tracker, tracks
        , refer_time, conf=obj_cfg['yolo']['conf'])
    thread1 = MyThread(grab_screen, args=())
    thread1.start()
    #im1, bev_lanes = clrnet.forward(cv2.resize(img, (1280, 720)), nav_line, im1, CAM)  # 传入RGB图像
    # 车辆前向物体检测线(暂时替代车道线)
    bev_lanes = FCW(nav_line, truck.speed)
    thread1.join()
    img, im0 = thread1.get_result()
    bar = cv2.cvtColor(img[750:768, 545:595, :], cv2.COLOR_RGB2BGR)  # 截取信息条[18, 480, 3]

    pool = ThreadPoolExecutor(max_workers=3)
    thread2 = pool.submit(cipv_notice, obstacles, bev_lanes)
    thread3 = pool.submit(line_filter, cv2.resize(img, (1280, 720)), M, traffic_light)
    thread4 = pool.submit(nav_process, cv2.cvtColor(img[610:740, 580:780, :], cv2.COLOR_RGB2BGR), nav_line, info, cipv)

    if time.time() - weather_time >= 60:
        weather, scene = weather_infer(img, weather_classifier)
        weather_time = time.time()

    obstacles, cipv = thread2.result()
    stop_line = thread3.result()
    nav_line, curve_speed_limit = thread4.result()
    pool.shutdown()

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
    truck = speed_detect(ocr, bar, truck)
    speed_limit = curve_speed_limit


    # 障碍物及停止线检测
    obstacles, cipv = cipv_notice(obstacles, bev_lanes)
    if traffic_light is not None:
        if traffic_light.tlcolor == 1 or traffic_light.tlcolor == 2:  # 检测到红灯或黄灯时再检测停止线
            stop_line = line_filter(cv2.resize(img, (1280, 720)), M, traffic_light)
        else:
            stop_line = None
    else:
        stop_line = None

    # 决策
    if obstacles is not None:
        state_trigger = planetrigger.update_trigger(fsmplanner.state, obstacles, bev_lanes, info, truck)
        if state_trigger is not None:
            fsmplanner.trigger(state_trigger)
            state = fsmplanner.state

    # 控制
    acc = 0.5
    ang = 0.5

    # 路口停车
    if traffic_light is not None:
        if cipv is not None:
            stop_distance = min(traffic_light.distance, 470 - int(cipv.position[1] + cipv.position[3]))
        else:
            stop_distance = traffic_light.distance
        if stop_distance <= 40:
            intersection_condition = 1  # 接近路口
        if traffic_light.tlcolor == 1 and 2:
            intersection_condition = 2  # 减速接近状态
            if stop_distance <= 30 and intersection_condition in [1, 2]:
                intersection_condition = 3  # 刹车状态
            elif intersection_condition == 3 and truck.speed == 0:
                intersection_condition = 4  # 停车状态

        elif traffic_light.tlcolor == 0 and intersection_condition != 1:
            intersection_condition = 0
    else:
        intersection_condition = 0

    if intersection_condition == 1:
        speed_limit = min(speed_limit, 40)
    elif intersection_condition == 2:
        speed_limit = min(speed_limit, 25)
    elif intersection_condition == 3:
        speed_limit = 0

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

    if intersection_condition == 4:
        acc = 0.501

    if info.activeAP:
        driver(ang, acc)
    elif info.AP_exit_reason == 1:
        driver(0.5, 0.75)
    else:
        driver(0.5, 0.5)

    truck.update(ang, acc, refer_time)

    # BEV绘制
    bevmap = draw_bev(nav_line, obstacles, traffic_light, bev_lanes, stop_line)
    bevmap = bevmap[160:, 160:640, :]
    bevmap = cv2.resize(bevmap, (416, 346))
    bevmap = print_info(bevmap, refer_time, truck, speed_limit, state, weather, scene, timeofday, info, planetrigger)

    img_show = cv2.vconcat([cv2.resize(im1, (416, 234)), bevmap])
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
