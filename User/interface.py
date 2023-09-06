import os
import sys
import numpy as np
import cv2
import math

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from User.draw import draw_line_3d, draw_box_3d
from Camera.transform import Calibration


class UserInterface:
    def __init__(self):
        self.speed = 0
        self.overspeed = False
        self.mode = 0
        self.mode_dic = {0: 'Manual', 1: 'Lat-Control', 2: 'Long-Control', 3: 'AP'}
        self.desire = 0
        self.desire_dic = {0: 'Direct', 1: 'True Left', 2: 'Turn Right', 3: 'Left LC', 4: 'Right LC'}
        rpy_calib_pred = np.array([0.0, 0.8, 0.0]) / 2
        self.show_size = (360, 360)
        self.calibration = Calibration(rpy_calib_pred, plot_img_width=self.show_size[0], plot_img_height=self.show_size[1])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.off = [1, 0, 15]
        self.objects_scale = [[3.0, 2,  4.5],  # car
                              [4.0, 3.0, 11.0],  # bus
                              [4.0, 2.9, 12.2]]  # truck

    def update(self, option_dict, truck):
        if truck is not None:
            self.speed = truck['speed']
            self.overspeed = truck['overspeed']
            self.mode = option_dict['mode']
            self.desire = option_dict['desire']

    def show(self, line_l, line_r, line_ll, line_rr, trajectory, dets_dict, fcw_state, option_dict, truck, fps):
        canva = np.ones([self.show_size[1], self.show_size[0], 3]) * 100
        self.update(option_dict, truck)
        
        pts_x = np.linspace(0, 40, 21)
        pts_z_y = np.zeros((21, 1))
        if self.desire not in [1, 2]:
            if line_l is not None and line_r is not None:
                fit_l = np.polyfit(line_l[:, 0], line_l[:, 1], 3)
                fit_r = np.polyfit(line_r[:, 0], line_r[:, 1], 3)
                pts_l_y = fit_l[0] * pts_x ** 3 + fit_l[1] * pts_x ** 2 + fit_l[2] * pts_x + fit_l[3]
                pts_r_y = fit_r[0] * pts_x ** 3 + fit_r[1] * pts_x ** 2 + fit_r[2] * pts_x + fit_r[3]
                line_l = np.concatenate((pts_x.reshape((-1, 1)), pts_l_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
                line_r = np.concatenate((pts_x.reshape((-1, 1)), pts_r_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
            
                canva = draw_line_3d(canva, line_l, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
                canva = draw_line_3d(canva, line_r, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
            if line_ll is not None:
                fit_ll = np.polyfit(line_ll[:, 0], line_ll[:, 1], 3)
                pts_ll_y = fit_ll[0] * pts_x ** 3 + fit_ll[1] * pts_x ** 2 + fit_ll[2] * pts_x + fit_ll[3]
                line_ll = np.concatenate((pts_x.reshape((-1, 1)), pts_ll_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
                canva = draw_line_3d(canva, line_ll, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
            if line_rr is not None:
                fit_rr = np.polyfit(line_rr[:, 0], line_rr[:, 1], 3)
                pts_rr_y = fit_rr[0] * pts_x ** 3 + fit_rr[1] * pts_x ** 2 + fit_rr[2] * pts_x + fit_rr[3]
                line_rr = np.concatenate((pts_x.reshape((-1, 1)), pts_rr_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
                canva = draw_line_3d(canva, line_rr, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)


        if trajectory is not None:
            pts_x = np.linspace(0, math.ceil(trajectory[-1, 0]), 21)
            fit_m = np.polyfit(trajectory[:, 0], trajectory[:, 1], 3)
            pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
            trajectory = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
            
            canva = draw_line_3d(canva, trajectory, self.calibration, 1.5, self.off, fill_color=(225, 238, 160))
        
        if dets_dict is not None:
            objects = dets_dict['objects']
            if objects is not None:
                for i in range(len(objects)):
                    cls_id, position = objects[i, 2], objects[i, :2]
                    theta = 0
                    position = np.concatenate((position, np.zeros((1))), axis=0)
                    canva = draw_box_3d(canva, position, theta, self.objects_scale[int(cls_id)], self.calibration, 0.1, self.off, color=(173, 202, 25))

        size = cv2.getTextSize(str(int(abs(self.speed) * 3.6)), self.font, 1.5, thickness=5)
        if self.overspeed == False:
            speed_color = (225, 238, 160)
        else:
            speed_color = (108, 96, 244)
        canva = cv2.putText(canva, str(abs(int(self.speed * 3.6))),
                            (int(self.show_size[0] / 2 - size[0][0] / 2), int(50 + size[0][1] / 2)), self.font, 1.5,
                            speed_color, 5)

        canva = cv2.putText(canva, self.mode_dic[self.mode], (10, 25), self.font, 0.5, (199, 237, 190), 2)
        canva = cv2.putText(canva, self.desire_dic[self.desire], (10, 50), self.font, 0.5, (199, 237, 190), 2)
        canva = cv2.putText(canva, str(fps), (self.show_size[0] - 50, 25), self.font, 0.5, (199, 237, 190), 2)

        if fcw_state == True:
            alpha = 0.9
            fcw_info = 'To Close'
            size = cv2.getTextSize(fcw_info, self.font, 2, thickness=6)
            warning_canvas = np.ones([self.show_size[1], self.show_size[0], 3]) * 200
            warning_canvas[int(self.show_size[1]/3):int(self.show_size[1]/3*2), :, :] = np.ones([int(self.show_size[1]/3), self.show_size[0], 3]) * np.array([108, 96, 244])
            warning_canvas = cv2.putText(warning_canvas, fcw_info, (int(self.show_size[0] / 2 - size[0][0] / 2), int(self.show_size[1]/2 + size[0][1] / 2)), self.font, 2,
                            (255, 255, 255), 6)
            canva = cv2.addWeighted(warning_canvas, alpha, canva, 1 - alpha, 0)

        return np.uint8(canva)


class DevInterface:
    def __init__(self):
        self.speed = 0
        self.mode = 0
        self.mode_dic = {0: 'Manual', 1: 'Lat-Control', 2: 'Long-Control', 3: 'AP'}
        self.desire = 0
        self.desire_dic = {0: 'Direct', 1: 'True Left', 2: 'Turn Right', 3: 'Left LC', 4: 'Right LC'}
        self.K = 10  # 放大系数
        self.shape = (30 * self.K, 12 * self.K)
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def update(self, option_dict, truck):
        if truck is not None:
            self.speed = truck['speed']
        self.mode = option_dict['mode']
        self.desire = option_dict['desire']

    def show(self, line_l, line_r, trajectory, option_dict, truck, fps):
        self.update(option_dict, truck)
        canva = np.ones([self.shape[0], self.shape[1], 3]) * 100

        if line_l is not None and line_r is not None:
            pts_x = np.linspace(0, 30, 16)
            fit_l = np.polyfit(line_l[:, 0], line_l[:, 1], 3)
            fit_r = np.polyfit(line_r[:, 0], line_r[:, 1], 3)
            pts_l_y = fit_l[0] * pts_x ** 3 + fit_l[1] * pts_x ** 2 + fit_l[2] * pts_x + fit_l[3]
            pts_r_y = fit_r[0] * pts_x ** 3 + fit_r[1] * pts_x ** 2 + fit_r[2] * pts_x + fit_r[3]
            line_l = np.concatenate((pts_x.reshape((-1, 1)), pts_l_y.reshape((-1, 1))), axis=1)
            line_r = np.concatenate((pts_x.reshape((-1, 1)), pts_r_y.reshape((-1, 1))), axis=1)
            
            for i in range(15):
                cv2.line(canva,
                         (int(line_l[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_l[i][0] * self.K)),
                         (int(line_l[i+1][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_l[i+1][0] * self.K)),
                         thickness=2,
                         color=(200, 200, 200))
            for i in range(15):
                cv2.line(canva,
                         (int(line_r[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_r[i][0] * self.K)),
                         (int(line_r[i+1][1] * self.K + self.shape[1] // 2), int(self.shape[0] - line_r[i+1][0] * self.K)),
                         thickness=2,
                         color=(200, 200, 200))
        if trajectory is not None:
            #pts_x = np.linspace(0, math.ceil(line_m[-1, 0]), 16)
            #fit_m = np.polyfit(line_m[:, 0], line_m[:, 1], 3)
            #pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
            #line_m = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1))), axis=1)
            for i in range(len(trajectory)-1):
                cv2.line(canva,
                         (int(trajectory[i][1] * self.K + self.shape[1] // 2), int(self.shape[0] - trajectory[i][0] * self.K)),
                         (int(trajectory[i+1][1] * self.K + self.shape[1] // 2), int(self.shape[0] - trajectory[i+1][0] * self.K)),
                         thickness=5,
                         color=(173, 202, 25))

        # canva = cv2.circle(canva, (int(self.shape[0] / 2), 50), 30, (200, 200, 200), thickness=5)
        size = cv2.getTextSize(str(int(self.speed * 3.6)), self.font, 1, thickness=3)
        canva = cv2.putText(canva, str(int(self.speed * 3.6)),
                            (int(self.shape[1] / 2 - size[0][0] / 2), int(30 + size[0][1] / 2)), self.font, 1,
                            (200, 200, 200), 3)

        canva = cv2.putText(canva, self.mode_dic[self.mode], (10, 25), self.font, 0.5, (0, 250, 0), 2)
        canva = cv2.putText(canva, self.desire_dic[self.desire], (10, 50), self.font, 0.5, (0, 250, 0), 2)
        canva = cv2.putText(canva, str(fps), (self.shape[0] - 50, 25), self.font, 0.5, (0, 250, 0), 2)
        
        # canva = np.hstack((info_canva, scene_canva))
        return np.uint8(canva)
