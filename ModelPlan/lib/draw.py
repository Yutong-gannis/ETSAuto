import numpy as np
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from transform import project_path
import cv2

X_IDXs = [
    0., 0.1875, 0.75, 1.6875, 3., 4.6875,
    6.75, 9.1875, 12., 15.1875, 18.75, 22.6875,
    27., 31.6875, 36.75, 42.1875, 48., 54.1875,
    60.75, 67.6875, 75., 82.6875, 90.75, 99.1875,
    108., 117.1875, 126.75, 136.6875, 147., 157.6875,
    168.75, 180.1875, 192.]


def draw_path(lane_lines, road_edges, path_plan, img_plot, calibration, lane_line_color_list, width=0.2, height=1.22,
              fill_color=(128, 0, 255), line_color=(0, 255, 0)):
    """Draw model predictions on an image."""

    overlay = img_plot.copy()
    alpha = 0.7
    fixed_distances = np.array(X_IDXs)[:, np.newaxis]

    # lane_lines are sequentially parsed ::--> means--> std's
    if lane_lines is not None:
        for line in lane_lines:
            line_l = line - np.array([0, 0.05, 0])
            line_r = line + np.array([0, 0.05, 0])
            img_pts_l = project_path(line_l, calibration, z_off=height)
            img_pts_r = project_path(line_r, calibration, z_off=height)

            for i in range(1, len(img_pts_l)):
                if i >= len(img_pts_r):
                    break

                u1, v1, u2, v2 = np.append(img_pts_l[i - 1], img_pts_r[i - 1])
                u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
                pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], fill_color)

    # road edges
    if road_edges is not None:
        (left_road_edge, right_road_edge), _ = road_edges

        calib_pts_ledg = np.hstack((fixed_distances, left_road_edge))
        calib_pts_redg = np.hstack((fixed_distances, right_road_edge))

        img_pts_ledg = project_path(calib_pts_ledg, calibration, z_off=0).reshape(-1, 1, 2)
        img_pts_redg = project_path(calib_pts_redg, calibration, z_off=0).reshape(-1, 1, 2)

        # plot road_edges
        cv2.polylines(overlay, [img_pts_ledg], False, (255, 128, 0), thickness=1)
        cv2.polylines(overlay, [img_pts_redg], False, (255, 234, 0), thickness=1)

    # path plan
    if path_plan is not None:

        path_plan_l = path_plan - np.array([0, width, 0])
        path_plan_r = path_plan + np.array([0, width, 0])

        img_pts_l = project_path(path_plan_l, calibration, z_off=height)
        img_pts_r = project_path(path_plan_r, calibration, z_off=height)

        for i in range(1, len(img_pts_l)):
            if i >= len(img_pts_r):
                break

            u1, v1, u2, v2 = np.append(img_pts_l[i - 1], img_pts_r[i - 1])
            u3, v3, u4, v4 = np.append(img_pts_l[i], img_pts_r[i])
            pts = np.array([[u1, v1], [u2, v2], [u4, v4], [u3, v3]], np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts], fill_color)
            # cv2.polylines(overlay, [pts], True, line_color)

    # drawing the plots on original iamge
    img_plot = cv2.addWeighted(overlay, alpha, img_plot, 1 - alpha, 0)

    return img_plot
