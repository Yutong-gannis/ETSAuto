import os
import sys
import numpy as np
import cv2

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

    def update(self, option_list, truck):
        self.speed = truck.speed
        self.mode = option_list[0]
        self.desire = option_list[1]

    def show(self, canve, lane_lines, path_plan, edge_lines, leaders, option_list, truck, fps):
        # canve = cv2.resize(canve, FULL_FRAME_SIZE, interpolation=cv2.INTER_AREA)
        canve = cv2.cvtColor(canve, cv2.COLOR_BGR2RGB)
        # canve = create_image_canvas(canve, CALIB_BB_TO_FULL, plot_img_height,
        #                             plot_img_width)
        self.update(option_list, truck)
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

    def update(self, option_list, truck):
        self.speed = truck['speed']
        self.mode = option_list[0]
        self.desire = option_list[1]

    def show(self, line_l, line_r, line_m, option_list, truck, fps):
        self.update(option_list, truck)
        canva = np.zeros([self.shape[0], self.shape[1], 3])

        if line_l is not None and line_r is not None:
            for i in range(len(line_l)):
                cv2.circle(canva,
                           (int(line_l[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_l[i][0] * self.K)),
                           radius=1,
                           thickness=-1,
                           color=(255, 255, 255))
            for i in range(len(line_r)):
                cv2.circle(canva,
                           (
                           int(line_r[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_r[i][0] * self.K)),
                           radius=1,
                           thickness=-1,
                           color=(255, 255, 255))
        if line_m is not None:
            for i in range(len(line_m)):
                cv2.circle(canva,
                           (int(line_m[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_m[i][0] * self.K)),
                           radius=1,
                           thickness=-1,
                           color=(100, 100, 100))

        # canva = cv2.circle(canva, (int(self.shape[0] / 2), 50), 30, (200, 200, 200), thickness=5)
        size = cv2.getTextSize(str(int(self.speed * 3.6)), self.font, 1, thickness=3)
        canva = cv2.putText(canva, str(int(self.speed * 3.6)),
                            (int(self.shape[1] / 2 - size[0][0] / 2), int(50 + size[0][1] / 2)), self.font, 1,
                            (200, 200, 200), 3)

        canva = cv2.putText(canva, self.mode_dic[self.mode], (10, 25), self.font, 0.5, (0, 250, 0), 2)
        canva = cv2.putText(canva, self.desire_dic[self.desire], (10, 50), self.font, 0.5, (0, 250, 0), 2)
        canva = cv2.putText(canva, str(fps), (self.shape[0] - 50, 25), self.font, 0.5, (0, 250, 0), 2)

        return canva
