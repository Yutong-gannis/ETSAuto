import numpy as np
import math


def horizontal_rounding(line):
    ptx_start = math.ceil(min(line[:, 0]))  # x轴取整
    ptx_end = math.floor(max(line[:, 0]))
    if ptx_end > 50:
        ptx_end = 50
    pts_x = np.linspace(ptx_start, ptx_end, (ptx_end - ptx_start) * 2 + 1)
    fit = np.polyfit(line[:, 0], line[:, 1], 3)
    pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
    line_rounding = np.concatenate((pts_x.reshape((-1, 1)), pts_y.reshape((-1, 1))), axis=1)
    return line_rounding


def get_skeleton(line_l, line_r):
    line_m = None
    if line_l is not None:
        line_m = (line_l + line_r) / 2
        line_m = np.concatenate((line_m, np.array([[-1, 0]])), axis=0)

        pts_x = np.linspace(0, 30, 30 * 2 + 1)
        fit = np.polyfit(line_m[:, 0], line_m[:, 1], 3)
        pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
        line_m = np.concatenate((pts_x.reshape((-1, 1)), pts_y.reshape((-1, 1))), axis=1)
    return line_m
