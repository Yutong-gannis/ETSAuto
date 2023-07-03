import numpy as np
import math
import cvxpy


class MPCPlanner:
    def __init__(self, truck):
        self.dt = truck.dt
        self.wheelbase = truck.wheelbase
        self.NX = 3  # x = x, y, yaw
        self.NU = 2  # u = [v,delta]
        self.T = 10  # horizon length
        self.R = np.diag([0.1, 0.1])  # input cost matrix
        self.Rd = np.diag([0.1, 0.1])  # input difference cost matrix
        self.Q = np.diag([1, 1, 1])  # state cost matrix
        self.Qf = self.Q  # state final matrix
        self.refer_path = np.zeros((10, 4))
        self.state = [0, 0, 0, 0]  # 车辆的状态(x,y,yaw,v)

        self.MAX_STEER = np.deg2rad(30.0)  # maximum steering angle [rad]
        self.MAX_DSTEER = np.deg2rad(5.0)  # maximum steering speed [rad/s]
        self.MAX_VEL = 22.0

    def update_path(self, plan_position, plan_yaw):
        if plan_position is not None:
            self.refer_path[:, 0:2] = plan_position[2:12, 0:2]
            self.refer_path[:, 2:3] = plan_yaw[2:12, 2:3]

            for i in range(len(self.refer_path)):
                if i == 0:
                    dx = self.refer_path[i + 1, 0] - self.refer_path[i, 0]
                    dy = self.refer_path[i + 1, 1] - self.refer_path[i, 1]
                    ddx = self.refer_path[2, 0] + self.refer_path[0, 0] - 2 * self.refer_path[1, 0]
                    ddy = self.refer_path[2, 1] + self.refer_path[0, 1] - 2 * self.refer_path[1, 1]
                elif i == (len(self.refer_path) - 1):
                    dx = self.refer_path[i, 0] - self.refer_path[i - 1, 0]
                    dy = self.refer_path[i, 1] - self.refer_path[i - 1, 1]
                    ddx = self.refer_path[i, 0] + self.refer_path[i - 2, 0] - 2 * self.refer_path[i - 1, 0]
                    ddy = self.refer_path[i, 1] + self.refer_path[i - 2, 1] - 2 * self.refer_path[i - 1, 1]
                else:
                    dx = self.refer_path[i + 1, 0] - self.refer_path[i, 0]
                    dy = self.refer_path[i + 1, 1] - self.refer_path[i, 1]
                    ddx = self.refer_path[i + 1, 0] + self.refer_path[i - 1, 0] - 2 * self.refer_path[i, 0]
                    ddy = self.refer_path[i + 1, 1] + self.refer_path[i - 1, 1] - 2 * self.refer_path[i, 1]
                self.refer_path[i, 2] = math.atan2(dy, dx)  # yaw
                # 计算曲率:设曲线r(t) =(x(t),y(t)),则曲率k=(x'y" - x"y')/((x')^2 + (y')^2)^(3/2).
                # 参考：https://blog.csdn.net/weixin_46627433/article/details/123403726
                self.refer_path[i, 3] = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2) ** (3 / 2))

    def updata_state(self, truck):
        self.state[3] = truck.speed

    def normalize_angle(self, angle):  # 统一到[-pi,pi]
        while angle > np.pi:
            angle -= 2.0 * np.pi

        while angle < -np.pi:
            angle += 2.0 * np.pi

        return angle

    def calc_track_error(self, x, y):  # 计算跟踪误差
        # 寻找参考轨迹最近目标点
        d_x = [self.refer_path[i, 0] - x for i in range(len(self.refer_path))]
        d_y = [self.refer_path[i, 1] - y for i in range(len(self.refer_path))]
        d = [np.sqrt(d_x[i] ** 2 + d_y[i] ** 2) for i in range(len(d_x))]
        s = np.argmin(d)  # 最近目标点索引

        yaw = self.refer_path[s, 2]
        k = self.refer_path[s, 3]
        angle = self.normalize_angle(yaw - math.atan2(d_y[s], d_x[s]))
        e = d[s]  # 误差
        if angle < 0:
            e *= -1

        return e, k, yaw, s

    def calc_ref_trajectory(self, dl=1.0):  # 计算参考轨迹点
        e, k, ref_yaw, ind = self.calc_track_error(self.state[0], self.state[1])

        xref = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((self.NU, self.T))  # 参考控制量
        ncourse = len(self.refer_path)

        xref[0, 0] = self.refer_path[ind, 0]
        xref[1, 0] = self.refer_path[ind, 1]
        xref[2, 0] = self.refer_path[ind, 2]

        # 参考控制量[v,delta]
        ref_delta = math.atan2(self.wheelbase * k, 1)
        dref[0, :] = self.state[3]
        dref[1, :] = ref_delta

        travel = 0.0

        for i in range(self.T + 1):
            travel += abs(self.state[3]) * self.dt
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = self.refer_path[ind + dind, 0]
                xref[1, i] = self.refer_path[ind + dind, 1]
                xref[2, i] = self.refer_path[ind + dind, 2]

            else:
                xref[0, i] = self.refer_path[ncourse - 1, 0]
                xref[1, i] = self.refer_path[ncourse - 1, 1]
                xref[2, i] = self.refer_path[ncourse - 1, 2]

        return xref, ind, dref

    def linear_mpc_control(self, xref, x0, delta_ref, ugv):
        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0  # 代价函数
        constraints = []  # 约束条件

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t] - delta_ref[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(x[:, t] - xref[:, t], self.Q)

            A, B, C = ugv.state_space(delta_ref[1, t], xref[2, t])
            constraints += [x[:, t + 1] - xref[:, t + 1] == A @ (x[:, t] - xref[:, t]) + B @ (u[:, t] - delta_ref[:, t])]

        cost += cvxpy.quad_form(x[:, self.T] - xref[:, self.T], self.Qf)

        constraints += [(x[:, 0]) == x0]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_VEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            opt_x = self.get_nparray_from_matrix(x.value[0, :])
            opt_y = self.get_nparray_from_matrix(x.value[1, :])
            opt_yaw = self.get_nparray_from_matrix(x.value[2, :])
            opt_v = self.get_nparray_from_matrix(u.value[0, :])
            opt_delta = self.get_nparray_from_matrix(u.value[1, :]) / np.pi + 0.5
            opt_delta = opt_delta

        else:
            print("Error: Cannot solve mpc..")
            opt_v, opt_delta, opt_x, opt_y, opt_yaw = None, None, None, None, None,

        return opt_v, opt_delta, opt_x, opt_y, opt_yaw

    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()

    def optimization(self, plan_position, plan_yaw, ugv):
        self.update_path(plan_position, plan_yaw)
        self.updata_state(ugv)
        x0 = self.state[0:3]
        xref, ind, dref = self.calc_ref_trajectory()
        opt_v, opt_delta, opt_x, opt_y, opt_yaw = self.linear_mpc_control(xref, x0, dref, ugv)
        return opt_v, opt_delta, opt_x, opt_y

