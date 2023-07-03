import os
import sys
import numpy as np
import cv2

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../Camera'))
from draw import draw_path, draw_leaders
from transform import Calibration


class UserInterface:
    def __init__(self):
        self.speed = 0
        self.mode = 0
        self.mode_dic = {0: 'Manual', 1: 'Lateral Control', 2: 'Longitudinal Control', 3: 'AP'}
        self.desire = 0
        self.desire_dic = {0: 'Direct', 1: 'True Left', 2: 'Turn Right', 3: 'Left Lane Change', 4: 'Right Lane Change',
                           5: 'Keep Left', 6: 'Keep Right'}
        rpy_calib_pred = np.array([0.00018335809, 0.034165092, -0.014245722]) / 2
        self.calibration = Calibration(rpy_calib_pred, plot_img_width=1360, plot_img_height=768)
        self.show_size = (400, 240)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def update(self, option_list, truck):
        self.speed = truck.speed
        self.mode = option_list[0]
        self.desire = option_list[1]

    def show(self, canve, lane_lines, path_plan, edge_lines, leaders, option_list, truck, fps):
        self.update(option_list, truck)
        if self.desire in [1, 2, 3, 4]:
            lane_lines, edge_lines = None, None
        if self.mode == 3:
            canve = draw_path(canve, lane_lines, path_plan, edge_lines, self.calibration,
                              width=(0.5, 0.05), height=2, color=(0, 200, 0))
        else:
            canve = draw_path(canve, lane_lines, path_plan, edge_lines, self.calibration,
                              width=(0.5, 0.05), height=2, color=(200, 200, 200))
        canve = draw_leaders(canve, leaders, self.calibration, height=1.2, color=(0, 200, 200))

        # canve = cv2.resize(canve, (640, 360))[:, 120:520]
        canve = cv2.resize(canve, (480, 270))[:240, 40:440]
        canve = cv2.circle(canve, (int(self.show_size[0] / 2), 50), 30, (200, 200, 200), thickness=5)
        size = cv2.getTextSize(str(int(self.speed * 3.6)), self.font, 1, thickness=3)
        canve = cv2.putText(canve, str(int(self.speed * 3.6)),
                            (int(self.show_size[0] / 2 - size[0][0] / 2), int(50 + size[0][1] / 2)), self.font, 1,
                            (200, 200, 200), 3)

        canve = cv2.putText(canve, self.mode_dic[self.mode], (10, 25), self.font, 0.75, (0, 250, 0), 2)
        canve = cv2.putText(canve, self.desire_dic[self.desire], (10, 50), self.font, 0.75, (0, 250, 0), 2)
        canve = cv2.putText(canve, str(fps), (self.show_size[0]-50, 25), self.font, 0.5, (0, 250, 0), 2)
        cv2.imshow("user_interface", canve)


class DevInterface:
    def __init__(self):
        self.K = 4  # 放大倍数
        self.shape = (200*self.K, 50*self.K)

    def show(self, lanes, edges, plan_position):
        canve = np.zeros([self.shape[0], self.shape[1], 3])
        cv2.line(canve, (self.shape[1] // 2, 0), (self.shape[1]//2, self.shape[0]), [30, 30, 30], thickness=1)
        cv2.line(canve, (self.shape[1] // 4, 0), (self.shape[1] // 4, self.shape[0]), [30, 30, 30], thickness=1)
        cv2.line(canve, (int(self.shape[1] * 0.75), 0), (int(self.shape[1] * 0.75), self.shape[0]), [30, 30, 30], thickness=1)

        for i in range(10):
            cv2.line(canve, (0, self.shape[0] - 10 * i * self.K), (self.shape[1], self.shape[0] - 10 * i * self.K),
                     [30, 30, 30], thickness=1)

        if lanes is not None and len(lanes):
            for lane in lanes:
                for i in range(33):
                    cv2.circle(canve,
                               (int(lane[i][1] * self.K + self.shape[1]//2), int(self.shape[0] - lane[i][0] * self.K)),
                               radius=1,
                               thickness=-1,
                               color=(125, 125, 125))

        if edges is not None and len(edges):
            for edge in edges:
                for i in range(33):
                    cv2.circle(canve,
                               (int(edge[i][1] * self.K + self.shape[1]//2), int(self.shape[0] - edge[i][0] * self.K)),
                               radius=1,
                               thickness=-1,
                               color=(0, 0, 255))

        if plan_position is not None and len(plan_position):
            for i in range(1, 33):
                cv2.circle(canve,
                           (int(plan_position[i][1] * self.K + self.shape[1]//2), int(self.shape[0] - plan_position[i][0] * self.K)),
                           radius=1,
                           thickness=-1,
                           color=(0, 255, 0))
                cv2.line(canve,
                         (int(plan_position[i - 1][1] * self.K + self.shape[1]//2), int(self.shape[0] - plan_position[i - 1][0] * self.K)),
                         (int(plan_position[i][1] * self.K + self.shape[1]//2), int(self.shape[0] - plan_position[i][0] * self.K)),
                         [0, 255, 0],
                         thickness=1)

        cv2.imshow("dev_interface", canve)
