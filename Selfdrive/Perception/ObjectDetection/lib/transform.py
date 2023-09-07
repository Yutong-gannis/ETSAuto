import numpy as np


class Cam_Transform:
    """
    This is class to do coordinate transformation
    """
    def __init__(self):
        beta = np.pi/2  # rotation of camera
        self.R = np.array([[1, 0,   0],
                           [0, np.cos(beta), -np.sin(beta)],
                           [0, np.sin(beta), np.cos(beta)]])  # 机身坐标与地面坐标的转换矩阵
        self.T = [0, 1.2, 0] # 机身坐标与地面坐标间的平移向量
        self.f = [600, 700]
        self.center = [680, 384]
        self.camera_intrinsic = [[self.f[0],         0, self.center[0]],
                                 [        0, self.f[1], self.center[1]],
                                 [        0,         0,             1]] # 相机内参
        self.K_inv = np.mat(self.camera_intrinsic).I  # 相机内参的逆
        self.R_inv = np.mat(self.R).I
        self.t = np.asmatrix(self.T).T
        self.R_inv_T = np.dot(self.R_inv, np.asmatrix(self.t))

    def pixel_to_world(self, img_point):
        """
        THis is function to convert image coordinate to bev
        
        :param img_point: Points on the image
        :type img_point: np.array
        :return bev_point: points in bev
        :rtype bev_point: np.array
        """
        coords = np.hstack((img_point, np.ones((img_point.shape[0], 1)))).T # 像素坐标
        cam_point = np.dot(self.K_inv, coords) # 像素坐标->物理成像坐标
        cam_R_inv = np.dot(self.R_inv, cam_point) # 求zc
        scale = self.R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = (np.asmatrix(scale_world) - np.asmatrix(self.R_inv_T)).T
        bev_point = world_point[:, [1,0]]
        return bev_point