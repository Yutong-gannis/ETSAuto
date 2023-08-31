import numpy as np
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../Camera'))
from transform import project_path
import cv2


def draw_line_3d(canve, line, calibration, width=0.2, height=1.22, fill_color=(128, 0, 255)):
    line_l = line - np.array([0, width/2, 0])
    line_r = line + np.array([0, width/2, 0])
    img_pts_l = project_path(line_l, calibration, z_off=height)
    img_pts_r = project_path(line_r, calibration, z_off=height)

    for i in range(1, len(img_pts_l)):
        if i >= len(img_pts_r):
            break

        u1, v1, u2, v2 = np.append(img_pts_l[i - 1], img_pts_r[i - 1])
        u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
        pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canve, [pts], fill_color)
        # cv2.polylines(canve, [pts], True, (100, 100, 100))
    return canve


def draw_point_3d(canve, point, calibration, height=1.22, fill_color=(0, 200, 200)):
    point = project_path(point, calibration, z_off=height)
    if len(point):
        canve = cv2.circle(canve, (point[0, 0], point[0, 1]), 10, fill_color, thickness=-1)
    return canve


def draw_path(canve, lane_lines, path_plan, edge_lines, calibration, width=None, height=1.5, color=(0, 200, 0)):
    """
    :param canve:  画板
    :param lane_lines: 车道线三维点集
    :param path_plan: 引导线三维点集
    :param edge_lines: 道路边缘三维点击
    :param calibration: 相机参数
    :param width: 线宽
    :param height: 视角高度
    :param color: 引导线颜色
    :return: 可视化结果
    """
    if width is None:
        width = [0.3, 0.05]
    overlay = canve.copy()
    alpha = 0.7

    if lane_lines is not None:
        for line in lane_lines:
            overlay = draw_line_3d(overlay, line[:15, :], calibration, width[1], height, fill_color=(200, 200, 200))

    if edge_lines is not None:
        for line in edge_lines:
            overlay = draw_line_3d(overlay, line[:15, :], calibration, width[1], height, fill_color=(0, 0, 200))

    # path plan
    if path_plan is not None:
        overlay = draw_line_3d(overlay, path_plan[:15, 0, :], calibration, width[0], height, fill_color=color)

    # drawing the plots on original iamge
    img_plot = cv2.addWeighted(overlay, alpha, canve, 1 - alpha, 0)
    return img_plot


def draw_leaders(canve, leader, calibration, height=1.5, color=(0, 200, 200)):
    if leader is not None:
        canve = draw_point_3d(canve, leader[0:1, 0:3], calibration, height, fill_color=color)
    return canve


def create_image_canvas(img_rgb, zoom_matrix, plot_img_height, plot_img_width):
    """Transform with a correct warp/zoom transformation."""
    img_plot = np.zeros((plot_img_height, plot_img_width, 3), dtype='uint8')
    cv2.warpAffine(img_rgb, zoom_matrix[:2], (img_plot.shape[1], img_plot.shape[0]), dst=img_plot,
                   flags=cv2.WARP_INVERSE_MAP)
    return img_plot
