import numpy as np
import math
from scipy.optimize import curve_fit


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

        pts_x = np.linspace(0, 35, 35 * 2 + 1)
        fit = np.polyfit(line_m[:, 0], line_m[:, 1], 3)
        pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
        line_m = np.concatenate((pts_x.reshape((-1, 1)), pts_y.reshape((-1, 1))), axis=1)
    return line_m

def get_z_line(lines):
    lines_z = []
    for line in lines:
        if line is not None:
            lines_z.append(line[:, [0, 2]])
    if len(lines_z) >= 2: 
        line_z = lines_z[0]
        for i in range(len(lines_z) - 1):
            line_z = np.vstack((line_z, lines_z[i+1]))
    elif len(lines_z) == 1:
        line_z = lines_z[0]

    else:
        line_z = None
    
    def f(x, a, b, c):
        return a*x**3 + b*x**2 + c*x
    if line_z is not None:
        line_z = line_z[np.lexsort(line_z[:, ::-1].T)]

        line_z = np.concatenate((line_z, np.array([[0, 0], [3, 0], [-1, 0]])), axis=0)
        pts_x = np.linspace(0, math.floor(max(line_z[:, 0])), math.floor(max(line_z[:, 0])) * 2 + 1)
        fit, _ = curve_fit(f, line_z[:, 0], line_z[:, 1])
        # fit = np.polyfit(line_z[:, 0], line_z[:, 1], 3)
        pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x
        # pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
        line_z = np.concatenate((pts_x.reshape((-1, 1)), pts_y.reshape((-1, 1))), axis=1)
        return line_z