import os
import sys
import numpy as np
import cv2
import math

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../Camera'))
from draw import draw_path, draw_leaders, create_image_canvas
from transform import Calibration
from Camera.constant import CALIB_BB_TO_FULL, plot_img_width, plot_img_height, FULL_FRAME_SIZE


class UserInterface:
    def __init__(self):
        self.speed = 0
        self.mode = 0
        self.mode_dic = {0: 'Manual', 1: 'Lateral Control', 2: 'Longitudinal Control', 3: 'AP'}
        self.desire = 0
        self.desire_dic = {0: 'Direct', 1: 'True Left', 2: 'Turn Right', 3: 'Left Lane Change', 4: 'Right Lane Change',
                           5: 'Keep Left', 6: 'Keep Right'}
        rpy_calib_pred = np.array([0.0, 0.034165092, -0.014245722]) / 2
        self.calibration = Calibration(rpy_calib_pred, plot_img_width=1360, plot_img_height=768)
        self.show_size = (400, 240)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def update(self, option_dict, truck):
        self.speed = truck.speed
        self.mode = option_dict['mode']
        self.desire = option_dict['desire']

    def show(self, canve, lane_lines, path_plan, edge_lines, leaders, option_dict, truck, fps):
        # canve = cv2.resize(canve, FULL_FRAME_SIZE, interpolation=cv2.INTER_AREA)
        canve = cv2.cvtColor(canve, cv2.COLOR_BGR2RGB)
        # canve = create_image_canvas(canve, CALIB_BB_TO_FULL, plot_img_height,
        #                             plot_img_width)
        self.update(option_dict, truck)
        if self.desire in [1, 2, 3, 4]:
            lane_lines, edge_lines = None, None
        if self.mode == 3:
            canve = draw_path(canve, lane_lines, path_plan, edge_lines, self.calibration,
                              width=(0.5, 0.1), height=1.5, color=(0, 200, 0))
        else:
            canve = draw_path(canve, lane_lines, path_plan, edge_lines, self.calibration,
                              width=(0.5, 0.1), height=1.5, color=(200, 200, 200))
        canve = draw_leaders(canve, leaders, self.calibration, height=1.5, color=(0, 200, 200))

        # canve = cv2.resize(canve, (640, 360))[:, 120:520]
        canve = cv2.resize(canve, (480, 270))[:240, 40:440]
        canve = cv2.circle(canve, (int(self.show_size[0] / 2), 50), 30, (200, 200, 200), thickness=5)
        size = cv2.getTextSize(str(int(self.speed * 3.6)), self.font, 1, thickness=3)
        canve = cv2.putText(canve, str(int(self.speed * 3.6)),
                            (int(self.show_size[0] / 2 - size[0][0] / 2), int(50 + size[0][1] / 2)), self.font, 1,
                            (200, 200, 200), 3)

        canve = cv2.putText(canve, self.mode_dic[self.mode], (10, 25), self.font, 0.75, (0, 250, 0), 2)
        canve = cv2.putText(canve, self.desire_dic[self.desire], (10, 50), self.font, 0.75, (0, 250, 0), 2)
        canve = cv2.putText(canve, str(fps), (self.show_size[0] - 50, 25), self.font, 0.5, (0, 250, 0), 2)
        cv2.imshow("user_interface", canve)


class DevInterface:
    def __init__(self):
        self.speed = 0
        self.mode = 0
        self.mode_dic = {0: 'Manual', 1: 'Lateral Control', 2: 'Longitudinal Control', 3: 'AP'}
        self.desire = 0
        self.desire_dic = {0: 'Direct', 1: 'True Left', 2: 'Turn Right', 3: 'Left Lane Change', 4: 'Right Lane Change'}
        self.K = 6  # 放大系数
        self.shape = (50 * self.K, 20 * self.K)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def update(self, option_dict, truck):
        if truck is not None:
            self.speed = truck['speed']
        self.mode = option_dict['mode']
        self.desire = option_dict['desire']

    def show(self, line_l, line_r, line_m, option_dict, truck, fps):
        self.update(option_dict, truck)
        scene_canva = np.ones([self.shape[0], self.shape[1], 3]) * 100
        info_canva = np.ones([self.shape[0], self.shape[1], 3]) * 100

        if line_l is not None and line_r is not None:
            pts_x = np.linspace(0, 30, 16)
            fit_l = np.polyfit(line_l[:, 0], line_l[:, 1], 3)
            fit_r = np.polyfit(line_r[:, 0], line_r[:, 1], 3)
            pts_l_y = fit_l[0] * pts_x ** 3 + fit_l[1] * pts_x ** 2 + fit_l[2] * pts_x + fit_l[3]
            pts_r_y = fit_r[0] * pts_x ** 3 + fit_r[1] * pts_x ** 2 + fit_r[2] * pts_x + fit_r[3]
            line_l = np.concatenate((pts_x.reshape((-1, 1)), pts_l_y.reshape((-1, 1))), axis=1)
            line_r = np.concatenate((pts_x.reshape((-1, 1)), pts_r_y.reshape((-1, 1))), axis=1)
            
            for i in range(15):
                cv2.line(scene_canva,
                         (int(line_l[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_l[i][0] * self.K)),
                         (int(line_l[i+1][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_l[i+1][0] * self.K)),
                         thickness=2,
                         color=(200, 200, 200))
            for i in range(15):
                cv2.line(scene_canva,
                         (int(line_r[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_r[i][0] * self.K)),
                         (int(line_r[i+1][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_r[i+1][0] * self.K)),
                         thickness=2,
                         color=(200, 200, 200))
        if line_m is not None:
            #pts_x = np.linspace(0, math.ceil(line_m[-1, 0]), 16)
            #fit_m = np.polyfit(line_m[:, 0], line_m[:, 1], 3)
            #pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
            #line_m = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1))), axis=1)
            for i in range(len(line_m)-1):
                cv2.line(scene_canva,
                         (int(line_m[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_m[i][0] * self.K)),
                         (int(line_m[i+1][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_m[i+1][0] * self.K)),
                         thickness=5,
                         color=(173, 202, 25))

        # canva = cv2.circle(canva, (int(self.shape[0] / 2), 50), 30, (200, 200, 200), thickness=5)
        size = cv2.getTextSize(str(int(self.speed * 3.6)), self.font, 1, thickness=3)
        scene_canva = cv2.putText(scene_canva, str(int(self.speed * 3.6)),
                            (int(self.shape[1] / 2 - size[0][0] / 2), int(30 + size[0][1] / 2)), self.font, 1,
                            (200, 200, 200), 3)

        info_canva = cv2.putText(info_canva, self.mode_dic[self.mode], (10, 25), self.font, 0.5, (0, 250, 0), 2)
        info_canva = cv2.putText(info_canva, self.desire_dic[self.desire], (10, 50), self.font, 0.5, (0, 250, 0), 2)
        info_canva = cv2.putText(info_canva, str(fps), (self.shape[0] - 50, 25), self.font, 0.5, (0, 250, 0), 2)
        
        # canva = np.hstack((info_canva, scene_canva))
        return np.uint8(info_canva), np.uint8(scene_canva)
