import numpy as np
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from constant import eon_intrinsics, W
import orientation as orientation


# device/mesh : x->forward, y-> right, z->down
# view : x->right, y->down, z->forward
device_frame_from_view_frame = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T


def line_transform(path, calibration, off):
    """Projects paths from calibration space (model input/output) to image space."""

    x = path[:, 0] + off[0]
    y = path[:, 1] + off[1]
    z = path[:, 2] + off[2]
    pts = calibration.car_space_to_bb(x, y, z)
    pts[pts < 0] = np.nan
    valid = np.isfinite(pts).all(axis=1)
    pts = pts[valid].astype(int)

    return pts


class Calibration:
    device_frame_from_view_frame = np.array([
        [0., 0., 1.],
        [1., 0., 0.],
        [0., 1., 0.]
    ])

    def get_view_frame_from_calib_frame(self, roll, pitch, yaw, height):
        device_from_calib = orientation.rot_from_euler([roll, pitch, yaw])
        view_from_calib = self.device_frame_from_view_frame.T.dot(device_from_calib)
        return np.hstack((view_from_calib, [[0], [height], [0]]))

    def __init__(self, rpy, intrinsic=eon_intrinsics, plot_img_width=640, plot_img_height=480):
        self.intrinsic = intrinsic
        self.extrinsics_matrix = self.get_view_frame_from_calib_frame(rpy[0], rpy[1], rpy[2], 0)[:, :3]
        self.plot_img_width = plot_img_width
        self.plot_img_height = plot_img_height
        self.zoom = W / plot_img_width
        self.CALIB_BB_TO_FULL = np.asarray([
            [self.zoom, 0., 0.],
            [0., self.zoom, 0.],
            [0., 0., 1.]])

    def car_space_to_ff(self, x, y, z):
        car_space_projective = np.column_stack((x, y, z)).T
        ep = self.extrinsics_matrix.dot(car_space_projective)
        kep = self.intrinsic.dot(ep)
        # TODO: fix numerical instability (add 1e-16)
        # UPD: this turned out to slow things down a lot. How do we do it then?
        return (kep[:-1, :] / kep[-1, :]).T

    def car_space_to_bb(self, x, y, z):
        pts = self.car_space_to_ff(x, y, z)
        return pts / self.zoom


def get_view_frame_from_calib_frame(roll, pitch, yaw, height):
    device_from_calib = orientation.rot_from_euler([roll, pitch, yaw])
    view_from_calib = view_frame_from_device_frame.dot(device_from_calib)
    return np.hstack((view_from_calib, [[0], [height], [0]]))
