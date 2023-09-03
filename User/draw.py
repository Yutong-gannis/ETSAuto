import numpy as np
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Camera.transform import line_transform
import cv2


def draw_line_3d(canve, line, calibration, width=0.2, off=[0.0, 0.0, 1.2], fill_color=(128, 0, 255), disconnect=True):
    line_l = line - np.array([0, width/2, 0])
    line_r = line + np.array([0, width/2, 0])
    img_pts_l = line_transform(line_l, calibration, off)
    img_pts_r = line_transform(line_r, calibration, off)

    for i in range(1, len(img_pts_l)):
        if i >= len(img_pts_r):
            break

        u1, v1, u2, v2 = np.append(img_pts_l[i - 1], img_pts_r[i - 1])
        u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
        pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canve, [pts], fill_color)
        if disconnect:
            cv2.polylines(canve, [pts], True, (100, 100, 100))
    return canve


def draw_point_3d(canve, point, calibration, height=1.22, fill_color=(0, 200, 200)):
    point = project_path(point, calibration, z_off=height)
    if len(point):
        canve = cv2.circle(canve, (point[0, 0], point[0, 1]), 10, fill_color, thickness=-1)
    return canve


def draw_leaders(canve, leader, calibration, height=1.5, color=(0, 200, 200)):
    if leader is not None:
        canve = draw_point_3d(canve, leader[0:1, 0:3], calibration, height, fill_color=color)
    return canve
