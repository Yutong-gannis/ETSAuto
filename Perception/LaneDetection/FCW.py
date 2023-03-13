import numpy as np

class Bev_Lane:  # 封装车道线类
    def __init__(self, cls, position_type, fit, endpoint, startpoint, lanes_pts):
        self.cls = cls  # 车道线线型 虚线：0， 实线：1
        self.position_type = position_type
        ###############车道线类型################
        # 左边界： -4
        # 左边第三根车道线：-3
        # 左边第二根（左临近车道的左边界）：-2
        # 当前车道左边界：-1
        # 当前车道中心线：0
        # 当前车道右边界：1
        # 右边第二根（右临近车道的左边界）：2
        # 右边第三根车道线：3
        # 右边界：4
        # 未知车道线：5 未定义前全部设置为5
        #######################################
        self.fit = fit  # 三次函数拟合车道线的参数 [a, b, c] 纵是X轴，横是Y轴
        self.endpoint = endpoint  # 远离本车的端点坐标
        self.startpoint = startpoint  # 接近本车的端点坐标
        self.pts = lanes_pts  # 车道线上所有的点

def FCW(nav_line, speed):
    long = 430 * (1 - speed / 1000 * 10)
    if long < 210:
        long = 210

    bev_lanes = []
    pts_y = np.linspace(long, 470, 20)
    pts_x_L10 = nav_line.fit[0] * pts_y ** 2 + nav_line.fit[1] * pts_y + nav_line.fit[2] - 10
    pts_x_R10 = nav_line.fit[0] * pts_y ** 2 + nav_line.fit[1] * pts_y + nav_line.fit[2] + 10

    L10_nav_pts = []
    R10_nav_pts = []
    for i in range(len(pts_y)):
        L10_nav_pts.append([int(pts_x_L10[i]), int(pts_y[i])])
        R10_nav_pts.append([int(pts_x_R10[i]), int(pts_y[i])])

    fit_L10 = np.polyfit(np.array([i[1] for i in L10_nav_pts]), np.array([i[0] for i in L10_nav_pts]), 3)
    fit_R10 = np.polyfit(np.array([i[1] for i in R10_nav_pts]), np.array([i[0] for i in R10_nav_pts]), 3)

    bev_lanes.append(Bev_Lane(1, -1, fit_L10, (pts_x_L10, pts_y[-1]), (pts_x_L10, pts_y[0]), L10_nav_pts))
    bev_lanes.append(Bev_Lane(1, 1, fit_R10, (pts_x_R10, pts_y[-1]), (pts_x_R10, pts_y[0]), R10_nav_pts))
    return bev_lanes