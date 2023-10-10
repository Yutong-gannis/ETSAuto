import numpy as np
import cv2

from lib.virtualcamera.transform import line_transform


def draw_line_3d(canva, line, calibration, width=0.2, off=[0.0, 0.0, 1.2], fill_color=(128, 0, 255), disconnect=True):
    """The function to draw line in 3d space

    :param canva: Cnava to draw
    :type canva: np.array
    :param line: 3D points on Line
    :type line: np.array
    :param calibration: Calibration of virtual camera
    :type calibration: Calibration
    :param width: Width of line, defaults to 0.2
    :type width: float, optional
    :param off: camera's offset, defaults to [0.0, 0.0, 1.2]
    :type off: list, optional
    :param fill_color: Color of line, defaults to (128, 0, 255)
    :type fill_color: tuple, optional
    :param disconnect: If cut line , defaults to True
    :type disconnect: bool, optional
    :return: Canva already draw
    :rtype: np.array
    """
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
        cv2.fillPoly(canva, [pts], fill_color)
        if disconnect:
            cv2.polylines(canva, [pts], True, (100, 100, 100))
    return canva


def draw_box_3d(canva, object, theta, object_scale, calibration, width=0.2, off=[0.0, 0.0, 1.2], color=(0, 200, 200)):
    """The function to draw 3d box

    :param canva: The draw canva
    :type canva: np.array
    :param object: Object to draw
    :type object: np.array
    :param theta: Box's rotation on axis y 
    :type theta: float
    :param object_scale: The scale of object
    :type object_scale: np.array
    :param calibration: Calibration of virtual camera
    :type calibration: Calibration
    :param width: The width of line, defaults to 0.2
    :type width: float, optional
    :param off: The offsets on 3 dims, defaults to [0.0, 0.0, 1.2]
    :type off: list, optional
    :param color: The color of box, defaults to (0, 200, 200)
    :type color: tuple, optional
    :return: The canva already draw
    :rtype: np.array
    """
    object = object[[0, 2, 1]]
    rotation_y = theta
    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    
    box_3d = compute_box_3d(object_scale, object, rotation_y)
    box_3d = box_3d[:, [0, 2, 1]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            line = np.array([box_3d[f[j], :], box_3d[f[(j + 1) % 4], :]])
            canva = draw_line_3d(canva, line, calibration, width, off, color, disconnect=None)
    return canva


def compute_box_3d(dim, location, rotation_y):
    """This is function to compute 8 corners of 3D box

    :param dim: The length, width, height of box
    :type dim: np.array
    :param location: The position of box
    :type location: np.array
    :param rotation_y: The ratation on y axis
    :type rotation_y: float
    :return: The 9 corners of box
    :rtype: np.array
    """
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l, l, 0, 0, l, l, 0, 0, l / 2]
    y_corners = [0, 0, 0, 0, h, h, h, h, h / 2]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

