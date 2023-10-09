import os
import sys
import numpy as np
import cv2
import math
from shared_memory_dict import SharedMemoryDict
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.insert(0, project_path)
from lib.draw import draw_line_3d, draw_box_3d
from lib.virtualcamera.transform import Calibration


class UserInterface:
    """Class to generate user's inference
    """
    def __init__(self):
        self.speed = 0
        self.overspeed = False
        self.fcw_state = False
        self.mode = 'manual'
        self.desire = 'Direct'
        rpy_calib_pred = np.array([0.0, 0.8, 0.0]) / 2
        self.show_size = (360, 360)
        self.calibration = Calibration(rpy_calib_pred, plot_img_width=self.show_size[0], plot_img_height=self.show_size[1])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.off = [1, 0, 15]
        self.objects_scale = [[3.0, 2,  4.5],  # car
                              [4.0, 3.0, 11.0],  # bus
                              [4.0, 2.9, 12.2]]  # truck
        self.line_l, self.line_r, self.line_ll, self.line_rr = None, None, None, None
        self.trajectory = None
        self.objects = None

    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
        lane_dict = SharedMemoryDict(name='lane', size=1024)
        dets_dict = SharedMemoryDict(name='dets', size=1024)
        condition_dict = SharedMemoryDict(name='condition', size=1024)
        fcw_dict = SharedMemoryDict(name='fcw', size=1024)
        plan_dict_sub = SharedMemoryDict(name='plan', size=1024)
        
        
        if 'trajectory' in plan_dict_sub.keys():
            self.trajectory = plan_dict_sub['trajectory']
        if 'fcw' in fcw_dict.keys():
            self.fcw_state = fcw_dict['fcw']
        if 'mode' in option_dict_sub.keys():
            self.mode = option_dict_sub['mode']
            self.desire = option_dict_sub['desire']
        if 'line_l' in lane_dict.keys():
            self.line_l, self.line_r, self.line_ll, self.line_rr = lane_dict['line_l'], lane_dict['line_r'], lane_dict['line_ll'], lane_dict['line_rr']
        if 'objects' in dets_dict.keys():
            self.objects = dets_dict['objects']
        if 'speed' in condition_dict.keys():
            self.speed = condition_dict['speed']
            self.overspeed = condition_dict['overspeed']

    def show(self):
        """Funtion to draw 3d information on opencv canva

        :return: Canva already draw
        :rtype: np.array
        """
        canva = np.ones([self.show_size[1], self.show_size[0], 3]) * 100
        
        pts_x = np.linspace(0, 60, 31)
        pts_z_y = np.zeros((31, 1))
        
        if self.desire not in [1, 2]:
            if self.line_l is not None and self.line_r is not None:
                fit_l = np.polyfit(self.line_l[:, 0], self.line_l[:, 1], 3)
                fit_r = np.polyfit(self.line_r[:, 0], self.line_r[:, 1], 3)
                pts_l_y = fit_l[0] * pts_x ** 3 + fit_l[1] * pts_x ** 2 + fit_l[2] * pts_x + fit_l[3]
                pts_r_y = fit_r[0] * pts_x ** 3 + fit_r[1] * pts_x ** 2 + fit_r[2] * pts_x + fit_r[3]
                line_l = np.concatenate((pts_x.reshape((-1, 1)), pts_l_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
                line_r = np.concatenate((pts_x.reshape((-1, 1)), pts_r_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
            
                canva = draw_line_3d(canva, line_l, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
                canva = draw_line_3d(canva, line_r, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
            if self.line_ll is not None:
                fit_ll = np.polyfit(self.line_ll[:, 0], self.line_ll[:, 1], 3)
                pts_ll_y = fit_ll[0] * pts_x ** 3 + fit_ll[1] * pts_x ** 2 + fit_ll[2] * pts_x + fit_ll[3]
                line_ll = np.concatenate((pts_x.reshape((-1, 1)), pts_ll_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
                canva = draw_line_3d(canva, line_ll, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
            if self.line_rr is not None:
                fit_rr = np.polyfit(self.line_rr[:, 0], self.line_rr[:, 1], 3)
                pts_rr_y = fit_rr[0] * pts_x ** 3 + fit_rr[1] * pts_x ** 2 + fit_rr[2] * pts_x + fit_rr[3]
                line_rr = np.concatenate((pts_x.reshape((-1, 1)), pts_rr_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
                canva = draw_line_3d(canva, line_rr, self.calibration, 0.2, self.off, fill_color=(200, 200, 200), disconnect=False)
        
        if self.trajectory is not None:
            pts_x = np.linspace(0, math.ceil(self.trajectory[-1, 0]), 31)
            fit_m = np.polyfit(self.trajectory[:, 0], self.trajectory[:, 1], 3)
            pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
            trajectory = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1)), pts_z_y.reshape((-1, 1))), axis=1)
            
            canva = draw_line_3d(canva, trajectory, self.calibration, 1.5, self.off, fill_color=(225, 238, 160))
        
         
        if self.objects is not None:
            for i in range(len(self.objects)):
                cls_id, position = self.objects[i, 2], self.objects[i, :2]
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

        canva = cv2.putText(canva, self.mode, (10, 25), self.font, 0.5, (199, 237, 190), 2)
        canva = cv2.putText(canva, self.desire, (10, 50), self.font, 0.5, (199, 237, 190), 2)

        if self.fcw_state == True:
            alpha = 0.9
            fcw_info = 'To Close'
            size = cv2.getTextSize(fcw_info, self.font, 2, thickness=6)
            warning_canvas = np.ones([self.show_size[1], self.show_size[0], 3]) * 200
            warning_canvas[int(self.show_size[1]/3):int(self.show_size[1]/3*2), :, :] = np.ones([int(self.show_size[1]/3), self.show_size[0], 3]) * np.array([108, 96, 244])
            warning_canvas = cv2.putText(warning_canvas, fcw_info, (int(self.show_size[0] / 2 - size[0][0] / 2), int(self.show_size[1]/2 + size[0][1] / 2)), self.font, 2,
                            (255, 255, 255), 6)
            canva = cv2.addWeighted(warning_canvas, alpha, canva, 1 - alpha, 0)

        return np.uint8(canva)
