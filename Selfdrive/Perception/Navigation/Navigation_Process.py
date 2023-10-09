import cv2
import os
import sys
import numpy as np
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.insert(0, project_path)
sys.path.insert(0, current_path)
from lib.filter import filter_out_red


class NavProcess:
    def __init__(self):
        self.length = 60
        pts_src = np.float32([
            [  0,   0], [200,   0],
            [  0, 130], [200, 130], ])

        pts_dst = np.float32([
            [  0,   0], [800,   0],
            [300, 600], [500, 600], ])
        self.h = cv2.getPerspectiveTransform(pts_src, pts_dst)
        
    def run(self, img):
        nav = cv2.cvtColor(img[610:740, 580:780, :], cv2.COLOR_RGB2BGR)
        bev_nav = self.nav2bev(nav)
        nav_line = self.get_nav_line(bev_nav)
        self.publish(nav_line)
        
    def nav2bev(self, map):  # 将导航地图转化为bev画布
        map = cv2.warpPerspective(map, self.h, (800, 600))
        map = filter_out_red(map)
        return map

    def get_nav_line(self, img):
        middle_pts = []
        x = 400
        i_start = 450
        for i in range(i_start, 330, -5):
            for j in range(x, 0, -1):
                if img[i, j] == 255:
                    break
            for k in range(x, 800, 1):
                if img[i, k] == 255:
                    break
            if j == 0 or j == 399 or k == 0 or k == 399:
                break
            x = int((j+k)/2)
            middle_pts.append([(i_start - i)/2, (x - 400)/2])
        
        middle_pts.append([-1, 0])
        middle_pts = np.array(middle_pts)  # 路线中心线
        if len(middle_pts) > 3:
            fit = np.polyfit(np.array([i[0] for i in middle_pts]), np.array([i[1] for i in middle_pts]), 3)
            pts_x = np.linspace(0, self.length, self.length * 2 + 1)
            pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
            # pts_y = fit[0] * pts_x ** 2 + fit[1] * pts_x + fit[2]
        else:
            pts_x = [0]
            pts_y = [0]

        nav_pts = []
        if len(pts_y):
            for pt_x, pt_y in zip(pts_x, pts_y):
                nav_pt = [pt_x, pt_y]
                nav_pts.append(nav_pt)
        nav_pts = np.array(nav_pts)
        return nav_pts
    
    def publish(self, nav_line):
        nav_dict_pub = SharedMemoryDict(name='nav', size=1024)
        nav_dict_pub["nav_line"] = nav_line

        
