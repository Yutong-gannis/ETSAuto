import matplotlib.pyplot as plt
import numpy as np
from math import *
from cvxopt import matrix, solvers


class MPC:
    def __init__(self):
        self.Np = 60  # 预测步长
        self.Nc = 60  # 控制步长

        self.dt = 0.1  # 时间间隔
        self.Length = 1.0  # 车辆轴距

        max_steer = 30 * pi / 180  # 最大方向盘打角
        max_steer_v = 15 * pi / 180  # 最大方向盘打角速度
        max_v = 8.7  # 最大车速
        max_a = 1.0  # 最大加速度

        # 目标函数相关矩阵
        self.Q = 50 * np.identity(3 * self.Np)  # 位姿权重
        self.R = 100 * np.identity(2 * self.Nc)  # 控制权重

        self.kesi = np.zeros((5, 1))

        self.A = np.identity(5)

        self.B = np.block([
            [np.zeros((3, 2))],
            [np.identity(2)]
        ])

        self.C = np.block([
            [np.identity(3), np.zeros((3, 2))]
        ])

        self.PHI = np.zeros((3 * self.Np, 5))
        self.THETA = np.zeros((3 * self.Np, 2 * self.Nc))

        self.CA = (self.Np + 1) * [self.C]

        self.H = np.zeros((2 * self.Nc, 2 * self.Nc))

        self.f = np.zeros((2 * self.Nc, 1))

        # 不等式约束相关矩阵
        A_t = np.zeros((self.Nc, self.Nc))
        for p in range(self.Nc):
            for q in range(p + 1):
                A_t[p][q] = 1

        A_I = np.kron(A_t, np.identity(2))

        # 控制量约束
        umin = np.array([[-max_v], [-max_steer]])
        umax = np.array([[max_v], [max_steer]])
        self.Umin = np.kron(np.ones((self.Nc, 1)), umin)
        self.Umax = np.kron(np.ones((self.Nc, 1)), umax)

        # 控制增量约束
        delta_umin = np.array([[-max_a * self.dt], [-max_steer_v * self.dt]])
        delta_umax = np.array([[max_a * self.dt], [max_steer_v * self.dt]])
        delta_Umin = np.kron(np.ones((self.Nc, 1)), delta_umin)
        delta_Umax = np.kron(np.ones((self.Nc, 1)), delta_umax)

        self.A_cons = np.zeros((2 * 2 * self.Nc, 2 * self.Nc))
        self.A_cons[0:2 * self.Nc, 0:2 * self.Nc] = A_I
        self.A_cons[2 * self.Nc:4 * self.Nc, 0:2 * self.Nc] = np.identity(2 * self.Nc)

        self.lb_cons = np.zeros((2 * 2 * self.Nc, 1))
        self.lb_cons[2 * self.Nc:4 * self.Nc, 0:1] = delta_Umin

        self.ub_cons = np.zeros((2 * 2 * self.Nc, 1))
        self.ub_cons[2 * self.Nc:4 * self.Nc, 0:1] = delta_Umax

    def mpcControl(self, x, y, yaw, v, angle, tar_x, tar_y, tar_yaw, tar_v, tar_angle):  # mpc优化控制
        T = self.dt
        L = self.Length

        # 更新误差
        self.kesi[0][0] = x - tar_x
        self.kesi[1][0] = y - tar_y
        self.kesi[2][0] = self.normalizeTheta(yaw - tar_yaw)
        self.kesi[3][0] = v - tar_v
        self.kesi[4][0] = angle - tar_angle

        # 更新A矩阵
        self.A[0][2] = -tar_v * sin(tar_yaw) * T
        self.A[0][3] = cos(tar_yaw) * T
        self.A[1][2] = tar_v * cos(tar_yaw) * T
        self.A[1][3] = sin(tar_yaw) * T
        self.A[2][3] = tan(tar_angle) * T / L
        self.A[2][4] = tar_v * T / (L * (cos(tar_angle) ** 2))

        # 更新B矩阵
        self.B[0][0] = cos(tar_yaw) * T
        self.B[1][0] = sin(tar_yaw) * T
        self.B[2][0] = tan(tar_angle) * T / L
        self.B[2][1] = tar_v * T / (L * (cos(tar_angle) ** 2))

        # 更新CA
        for i in range(1, self.Np + 1):
            self.CA[i] = np.dot(self.CA[i - 1], self.A)

        # 更新PHI和THETA
        for j in range(self.Np):
            self.PHI[3 * j:3 * (j + 1), 0:5] = self.CA[j + 1]
            for k in range(min(self.Nc, j + 1)):
                self.THETA[3 * j:3 * (j + 1), 2 * k: 2 * (k + 1)
                ] = np.dot(self.CA[j - k], self.B)

        # 更新H
        self.H = np.dot(np.dot(self.THETA.transpose(), self.Q),
                        self.THETA) + self.R

        # 更新f
        self.f = 2 * np.dot(np.dot(self.THETA.transpose(), self.Q),
                            np.dot(self.PHI, self.kesi))

        # 更新约束
        Ut = np.kron(np.ones((self.Nc, 1)), np.array([[v], [angle]]))
        self.lb_cons[0:2 * self.Nc, 0:1] = self.Umin - Ut
        self.ub_cons[0:2 * self.Nc, 0:1] = self.Umax - Ut

        # 求解QP
        P = matrix(self.H)
        q = matrix(self.f)
        G = matrix(np.block([
            [self.A_cons],
            [-self.A_cons]
        ]))
        h = matrix(np.block([
            [self.ub_cons],
            [-self.lb_cons]
        ]))

        solvers.options['show_progress'] = False
        sol = solvers.qp(P, q, G, h)
        X = sol['x']

        # 输出结果
        v += X[0]
        angle += X[1]

        return v, angle

    def normalizeTheta(self, angle):  # 角度归一化
        while (angle >= pi):
            angle -= 2 * pi

        while (angle < -pi):
            angle += 2 * pi

        return angle

    def findIdx(self, x, y, cx, cy):  # 寻找欧式距离最近的点
        min_dis = float('inf')
        idx = 0

        for i in range(len(cx)):
            dx = x - cx[i]
            dy = y - cy[i]
            dis = dx ** 2 + dy ** 2
            if (dis < min_dis):
                min_dis = dis
                idx = i

        return idx

    def update(self, x, y, yaw, v, angle):  # 模拟车辆位置
        x += v * cos(yaw) * self.dt
        y += v * sin(yaw) * self.dt
        yaw += v / self.Length * tan(angle) * self.dt

        return x, y, yaw

import time
if __name__ == '__main__':
    cx = np.linspace(0, 200, 20)
    cy = np.zeros(len(cx))
    dx = np.zeros(len(cx))
    ddx = np.zeros(len(cy))
    cyaw = np.zeros(len(cx))
    ck = np.zeros(len(cx))

    for i in range(len(cx)):
        cy[i] = cos(cx[i] / 10) * cx[i] / 10

    # 计算一阶导数
    for i in range(len(cx) - 1):
        dx[i] = (cy[i + 1] - cy[i]) / (cx[i + 1] - cx[i])
    dx[len(cx) - 1] = dx[len(cx) - 2]

    # 计算二阶导数
    for i in range(len(cx) - 2):
        ddx[i] = (cy[i + 2] - 2 * cy[i + 1] + cy[i]) / (0.5 * (cx[i + 2] - cx[i])) ** 2
    ddx[len(cx) - 2] = ddx[len(cx) - 3]
    ddx[len(cx) - 1] = ddx[len(cx) - 2]

    # 计算偏航角
    for i in range(len(cx)):
        cyaw[i] = atan(dx[i])

    # 计算曲率
    for i in range(len(cx)):
        ck[i] = ddx[i] / (1 + dx[i] ** 2) ** 1.5

    # 初始状态
    x = 0.0
    y = 5.0
    yaw = 0.0
    v = 0.0
    angle = 0.0
    t = 0

    # 历史状态
    xs = [x]
    ys = [y]
    vs = [v]
    angles = [angle]
    ts = [t]

    # 实例化
    mpc = MPC()

    while (1):
        time_start = time.time()
        idx = mpc.findIdx(x, y, cx, cy)
        if (idx == len(cx) - 1):
            break

        tar_v = 30.0 / 3.6
        tar_angle = atan(mpc.Length * ck[idx])

        (v, angle) = mpc.mpcControl(x, y, yaw, v, angle,
                                    cx[idx], cy[idx], cyaw[idx], tar_v, tar_angle)

        (x, y, yaw) = mpc.update(x, y, yaw, v, angle)
        print(time.time() - time_start)
        # 保存状态
        xs.append(x)
        ys.append(y)
        vs.append(v)
        angles.append(angle)
        t = t + 0.1
        ts.append(t)

        # 显示
        plt.plot(cx, cy)
        plt.scatter(xs, ys, c='r', marker='*')
        plt.pause(0.01)  # 暂停0.01秒
        plt.clf()

    plt.close()
    plt.subplot(2, 1, 1)
    plt.plot(ts, vs)
    plt.subplot(2, 1, 2)
    plt.plot(ts, angles)
    plt.show()