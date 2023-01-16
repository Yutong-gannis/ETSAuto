import numpy as np

class cam:
    def __init__(self):
        self.R = np.array([[1, 0,   0],
                           [0, 1, -11],
                           [0, 11, 1]])  # 机身坐标与地面坐标的转换矩阵
        self.T = [0, 0, 300] # 机身坐标与地面坐标间的平移向量
        self.f = [3000, 4000]
        self.center = [640, 0]
        '''
        # 已经调好的固定参数， 改坏后恢复
        self.R = [[1, 0, 0],
                  [0, 1, -11],
                  [0, 11, 1]]  # 机身坐标与地面坐标的转换矩阵
        self.T = [0, 0, 300]  # 机身坐标与地面坐标间的平移向量
        self.f = [3000, 4000]
        self.center = [640, 0]
        '''
        self.camera_intrinsic = [[self.f[0],         0, self.center[0]],
                                 [        0, self.f[1], self.center[1]],
                                 [        0,         0,             1]] # 相机内参
        self.K_inv = np.mat(self.camera_intrinsic).I  # 相机内参的逆
        self.R_inv = np.mat(self.R).I
        self.t = np.asmatrix(self.T).T
        self.R_inv_T = np.dot(self.R_inv, np.asmatrix(self.t))

    def update_intrinsic(self, vanish_point):
        R3 = vanish_point[1][0]*np.dot(self.K_inv, vanish_point)
        self.R[1, 2] = -R3[1][0]/3
        self.R[2, 1] = R3[1][0] / 3

    def pixel_to_world(self, img_point):
        coords = np.array([[img_point[0]], [img_point[1]], [1.0]]) # 像素坐标
        cam_point = np.dot(self.K_inv, coords) # 像素坐标->物理成像坐标
        cam_R_inv = np.dot(self.R_inv, cam_point) # 求zc
        scale = self.R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(self.R_inv_T)
        return world_point.T

class backcam_left:
    def __init__(self):
        self.R = [[1,  0,   0],
                  [0,  1,   0],
                  [0,  0,   1]] # 机身坐标与地面坐标的转换矩阵
        self.T = [0, 0, 100] # 机身坐标与地面坐标间的平移向量
        self.f = [1000, 1000]
        self.center = [180, 0]
        self.camera_intrinsic = [[self.f[0],         0, self.center[0]],
                                 [        0, self.f[1], self.center[1]],
                                 [        0,         0,             1]] # 相机内参
        self.K_inv = np.mat(self.camera_intrinsic).I # 相机内参的逆
        self.R_inv = np.mat(self.R).I
        self.t = np.asmatrix(self.T).T
        self.R_inv_T = np.dot(self.R_inv, np.asmatrix(self.t))

    def pixel_to_world(self, img_point):
        coords = np.array([[img_point[0]], [img_point[1]], [1.0]]) # 像素坐标
        cam_point = np.dot(self.K_inv, coords) # 像素坐标->物理成像坐标
        cam_R_inv = np.dot(self.R_inv, cam_point) # 求zc
        scale = self.R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(self.R_inv_T)
        return world_point.T

class backcam_right:
    def __init__(self):
        self.R = [[1, 0,   0],
                  [0, 1,   0],
                  [0, 0,   1]] # 机身坐标与地面坐标的转换矩阵
        self.T = [0, 0, 100] # 机身坐标与地面坐标间的平移向量
        self.f = [1000, 1000]
        self.center = [1100, 0]
        self.camera_intrinsic = [[self.f[0],         0, self.center[0]],
                                 [        0, self.f[1], self.center[1]],
                                 [        0,         0,             1]] # 相机内参
        self.K_inv = np.mat(self.camera_intrinsic).I # 相机内参的逆
        self.R_inv = np.mat(self.R).I
        self.t = np.asmatrix(self.T).T
        self.R_inv_T = np.dot(self.R_inv, np.asmatrix(self.t))

    def pixel_to_world(self, img_point):
        coords = np.array([[img_point[0]], [img_point[1]], [1.0]]) # 像素坐标
        cam_point = np.dot(self.K_inv, coords) # 像素坐标->物理成像坐标
        cam_R_inv = np.dot(self.R_inv, cam_point) # 求zc
        scale = self.R_inv_T[2][0] / cam_R_inv[2][0]
        scale_world = np.multiply(scale, cam_R_inv)
        world_point = np.asmatrix(scale_world) - np.asmatrix(self.R_inv_T)
        return world_point.T


def transposition(model, pts):
    output = model(pts)
    return output
