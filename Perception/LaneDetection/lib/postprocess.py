import numpy as np
import json
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline


def mean_col_by_row_with_offset_z(seg, offset_y, z):
    assert (len(seg.shape) == 2)

    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 一个id
        cols, rows, z_val = [], [], []
        for y_op in range(seg.shape[0]):  # Every row
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]  # All pos in this row
            z_op = z[y_op, :]
            offset_op = offset_y[y_op, :]
            if x_op.size > 0:
                offset_op = offset_op[x_op]
                z_op = np.mean(z_op[x_op])
                z_val.append(z_op)
                x_op_with_offset = x_op + offset_op
                x_op = np.mean(x_op_with_offset)  # mean pos
                cols.append(x_op)
                rows.append(y_op + 0.5)
        lines.append((cols, rows, z_val))
    return lines


def bev_instance2points_with_offset_z(ids: np.ndarray, max_x=50, meter_per_pixal=(0.2, 0.2), offset_y=None, Z=None):
    center = ids.shape[1] / 2
    lines = mean_col_by_row_with_offset_z(ids, offset_y, Z)
    points = []
    # for i in range(1, ids.max()):
    for y, x, z in lines:  # cols, rows
        # x, y = np.where(ids == 1)
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]
        z = np.array(z)[::-1]

        x = max_x / meter_per_pixal[0] - x
        y = y * meter_per_pixal[1]
        y -= center * meter_per_pixal[1]
        x = x * meter_per_pixal[0]

        # y *= -1.0  # Vector is from right to left
        if len(x) < 2:
            continue
        # spline = CubicSpline(x, y, extrapolate=False)
        points.append(np.concatenate((x.reshape(1, -1), y.reshape(1, -1), z.reshape(1, -1)), axis=0).T)
    return points


def mean_col_by_row(seg):
    assert (len(seg.shape) == 2)

    # optimize if already estimated
    curr_id = 0
    center_ids = np.unique(seg[seg > 0])
    lines = []
    for idx, cid in enumerate(center_ids):  # 一个id
        cols, rows, z_val = [], [], []

        for y_op in range(seg.shape[0]):  # Every row
            condition = seg[y_op, :] == cid
            x_op = np.where(condition)[0]  # All pos in this row
            if x_op.size > 0:
                x_op = np.mean(x_op)  # mean pos
                cols.append(4 * x_op)
                rows.append(4 * y_op)
        lines.append((cols, rows, cid))
    return lines


def bev_instance2points(ids: np.ndarray, max_x=50):
    center = ids.shape[1] / 2
    lines = mean_col_by_row(ids)
    points = []
    # for i in range(1, ids.max()):
    for y, x, cid in lines:  # cols, rows
        # x, y = np.where(ids == 1)
        x = np.array(x)[::-1]
        y = np.array(y)[::-1]

        if len(x) < 2:
            continue
        x = np.flip(x)
        y = np.flip(y)
        # c = interp1d(x, y, kind='cubic')
        f_p = CubicSpline(x, y)

        x_p = np.linspace(x[0], x[-1])
        y_p = f_p(x_p)
        points.append((y_p, x_p, cid))
    return lines, points


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x
