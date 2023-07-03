import numpy as np
import math
from scipy.optimize import curve_fit


# def a function
def f(x, a, b, c):
    return a * x ** 3 + b * x ** 2 + c * x


def Purepursuit(truck, nav_line):
    # fit, _ = curve_fit(f, np.array([i[1] for i in nav_line[0:10]]).astype(np.float32),
    #                    np.array([i[0] for i in nav_line[0:10]]).astype(np.float32))

    fit = np.polyfit(np.array([i[0] for i in nav_line[0:10]]).astype(np.float32),
                     np.array([i[1] for i in nav_line[0:10]]).astype(np.float32), 3)
    pts_y = np.linspace(0, 30, 30)
    pts_x = fit[0] * pts_y ** 3 + fit[1] * pts_y ** 2 + fit[2] * pts_y + fit[3]
    robot_state = np.zeros(2)  # 定义车辆位置
    robot_state[1] = truck.x
    robot_state[0] = truck.y - truck.wheelbase  # 后轴中心位置
    dy = np.average(pts_y[0:10]) - 0
    dx = np.average(pts_x[0:10]) - 0
    # dy, dx = np.average(nav_line[5:10], axis=0) - robot_state
    alpha = np.arctan(dx / dy)  # dx, dy夹角
    ang = math.atan2(2.0 * truck.wheelbase * np.sin(alpha), truck.ld) + 0.5  # pure pursuit controller
    return ang
