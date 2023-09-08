import numpy as np


def trans_translate(pts, dx, dy):
    T = [[1, 0, dx],
         [0, 1, dy],
         [0, 0, 1]]
    pts_1 = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))[:2, :].T
    return pts_1


def trans_rotate(pts, beta):
    # beta = np.deg2rad(beta)
    T = [[np.cos(beta), -np.sin(beta), 0],
         [np.sin(beta), np.cos(beta), 0],
         [0, 0, 1]]
    pts_1 = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))[:2, :].T
    return pts_1

def update_trajectory(trajectory, condition_dict):
    if condition_dict is not None:
          trajectory = trans_rotate(trajectory, -condition_dict['dtheta'])
          trajectory = trans_translate(trajectory, -condition_dict['dx'], -condition_dict['dy'])
    trajectory = trajectory[np.where(trajectory[:, 0] >= 0)[0], :]
    return trajectory
