import heapq
import numpy as np
from sklearn.cluster import DBSCAN
import cv2


class Driving_line:
    def __init__(self, pts, pts_x, pts_y):
        self.pts = pts
        self.pts_x = pts_x
        self.pts_y = pts_y


class Bev_Lane:  # 封装车道线类(金丹）
    def __init__(self, cls, position_type_nav, position_type_ego, fit, endpoint, startpoint, lanes_pts):
        self.cls = cls  # 车道线线型 虚线：0， 实线：1
        self.position_type_ego = position_type_ego
        ###############车道线类型################
        # 左边第二根（左临近车道的左边界）：-2
        # 当前车道左边界：-1
        # 当前车道中心线：0
        # 当前车道有边界：1
        # 右边第二根（左临近车道的左边界）：2
        # 位置：5
        #######################################
        self.position_type_nav = position_type_nav
        ###############车道线类型################
        # 导航车道右边界：1
        # 导航车道左边界（超车道右边界）：2
        # 超车道左边界：3
        # 未知：5
        #######################################
        self.fit = fit  # 三次函数拟合车道线的参数 [a, b, c] 纵是X轴，横是Y轴
        self.endpoint = endpoint  # 远离本车的端点坐标
        self.startpoint = startpoint  # 接近本车的端点坐标
        self.pts = lanes_pts  # 车道线上所有的点


def postprocess(img_, nav_line, output, CAM):
    lanes_xys = lane2lane_xys(output[0])
    lines_cls = line_classification(img_, lanes_xys)  # 车道线分类
    res, lanes = imshow_lanes(img_, lanes_xys)
    vanish_point = vanish_point_detection(lanes)  # 计算灭点
    bev_lanes = bev_lane(lanes, CAM, lines_cls, vanish_point)
    bev_lanes = line_positionegotype_classfication(bev_lanes)
    bev_lanes = line_positionnavtype_classfication(bev_lanes, nav_line)
    return res, bev_lanes


def lane2lane_xys(lanes):
    lanes = [lane.to_array() for lane in lanes]
    lanes_xys = []
    for _, lane in enumerate(lanes):
        xys = []
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            xys.append([x, y])
        if len(xys) > 0:
            lanes_xys.append(xys)
    lanes_xys.sort(key=lambda xys: xys[0][0])
    return lanes_xys


def imshow_lanes(img, lanes_xys, width=4):
    for idx, xys in enumerate(lanes_xys):
        for i in range(1, len(xys)):
            # cv2.line(img, xys[i - 1], xys[i], COLORS[idx], thickness=width)
            cv2.line(img, tuple(xys[i - 1]), tuple(xys[i]), [0, 255, 127], thickness=width)

    '''
    middle_lanes = []  # 计算中线
    if len(lanes_xys) >= 2:
        middle_lanes, middle_pt = middel_line(lanes_xys)
        for pt in middle_lanes:
            cv2.circle(img, pt, radius=3, color=(200, 200, 200), thickness=-1)
            # cv2.line(img, middle_pt, (middle_pt[0], middle_pt[1]-5), [0, 255, 127], thickness=width)
    '''
    return img, lanes_xys


def bev_lane(lanes, CAM, lines_cls, vanish_point):
    # CAM.update_intrinsic(vanish_point)  #根据灭点坐标的相机外参修正
    bev_lanes = []
    for i in range(len(lanes)):
        if len(lanes[i]) < 10:  # 过滤点数小于10的线
            continue
        bev_lane_pts = []
        bev_xy = [] # 拟合前
        for pt in lanes[i]:
            pt = list(pt)
            pt[1] = 720 - pt[1]  # 到摄像头像素坐标
            bev_pt = CAM.pixel_to_world(pt).tolist()
            if 470 - bev_pt[0][1]/2 <= 250 or 470 - bev_pt[0][1]/2 >= 500:
                continue
            bev_xy.append([int(bev_pt[0][0]/5)+400, 470 - int(bev_pt[0][1]/1.8)])
        bev_xy = np.array(bev_xy).T
        fit = np.polyfit(bev_xy[:][1], bev_xy[:][0], 3)  # 三次函数拟合
        bev_y = np.linspace(250, 470, 20)  # 为了减少计算量，采样点减少为20个
        bev_x = fit[0] * bev_y ** 3 + fit[1] * bev_y ** 2 + fit[2] * bev_y + fit[3]
        for j in range(20):
            if np.abs(bev_x[j]-400) >= 60: continue
            bev_lane_pts.append((int(bev_x[j]), int(bev_y[j])))
        if len(bev_lane_pts) >= 1: # 之前过滤了一次距离，可能全过滤掉了
            bev_line = Bev_Lane(lines_cls[i], 5, 5, fit, bev_lane_pts[-1], bev_lane_pts[0], bev_lane_pts)  # 封装bev车道线信息
            bev_lanes.append(bev_line)
    return bev_lanes


def vanish_point_detection(lanes): # 灭点检测
    if len(lanes) == 0:
        vanish_point = np.array([[640], [335], [0]])
    elif len(lanes) == 1:  # 只检测出一条车道时，只与中线求交
        lane = np.array(lanes[0])
        fit = np.polyfit(lane[:, 0], lane[:, 1], 1)
        vanish_point = np.array([[640], [fit[0] * 640 + fit[1]], [0]])
    elif len(lanes) > 1:  #检测出多条车道时， 两两求交
        vpts = []
        fits = []
        for lane in lanes:
            lane = np.array(lane)
            fit = np.polyfit(lane[:, 0], lane[:, 1], 1)
            fits.append(fit)
        for i in range(len(fits) - 1):
            for j in range(i+1, len(fits)):
                fit1 = fits[i]
                fit2 = fits[j]
                vpt_x = -(fit1[1] - fit2[1])/(fit1[0] - fit2[0])
                vpt_y = fit1[0] * vpt_x + fit1[1]
                vpts.append([vpt_x, vpt_y])
        vpts = np.array(vpts)
        vanish_point = np.array([[np.average(vpts[:, 0])], [np.average(vpts[:, 1])], [0]])
    return vanish_point


def middel_line(lanes_xys):
    line_to_middle = []
    middle_lane = []
    lanes = lanes_xys.copy()
    for i in range(len(lanes)-1, -1, -1):
        if len(lanes[i]) <= 30:
            del lanes[i]
    if len(lanes) >= 2:
        for i in range(len(lanes)):
            line_to_middle.append(np.abs(lanes[i][0][0]-640))
        current_lane_idx = list(map(line_to_middle.index, heapq.nsmallest(2, line_to_middle)))
        current_lane_0 = lanes[current_lane_idx[0]]
        current_lane_1 = lanes[current_lane_idx[1]]
        fit_0 = np.polyfit(np.array([i[1] for i in current_lane_0]), np.array([i[0] for i in current_lane_0]), 3)
        fit_1 = np.polyfit(np.array([i[1] for i in current_lane_1]), np.array([i[0] for i in current_lane_1]), 3)
        pts_y = np.linspace(500, 699, 50)
        pts_x = (fit_0[0]+fit_1[0])/2 * pts_y ** 3 + (fit_0[1]+fit_1[1])/2 * pts_y ** 2 + (fit_0[2]+fit_1[2])/2 * pts_y + (fit_0[3]+fit_1[3])/2
        for i in range(50):
            middle_lane.append((int(pts_x[i]), int(pts_y[i])))
        middle_pt = ((current_lane_0[0][0]+current_lane_1[0][0])//2, (current_lane_0[0][1]+current_lane_1[0][1])//2)
        return middle_lane, middle_pt
    elif len(lanes) == 1:
        if lanes[0][0][0] > 640:
            middle_lane = [(lanes[0][0][0]-550, lanes[0][0][1]), (lanes[0][0][0]-550, lanes[0][1][1]), (lanes[0][0][0]-550, lanes[0][2][1])]
        elif lanes[0][0][0] < 640:
            middle_lane = [(lanes[0][0][0]+550, lanes[0][0][1]), (lanes[0][0][0]-550, lanes[0][1][1]), (lanes[0][0][0]-550, lanes[0][2][1])]
        return middle_lane, []
    return middle_lane, []


def line_positionegotype_classfication(bev_lanes):
    positiontypes = np.zeros((1, len(bev_lanes)))  # 保存车道线位置类型
    bev_lanes_sorted = [0] * len(bev_lanes)
    if len(bev_lanes) >= 1:
        start_xs = []
        for bev_lane in bev_lanes:
            start_x = bev_lane.pts[-3][0]  # bev车道线起点横坐标
            start_xs.append(start_x-400)  # bev车道起点距本车中心线横向距离
        middle_lane_exist = False  # 判断是否存在车道线在车中间

        start_xs = np.array(start_xs) # 将车道线按起始点位置重新排序
        lane_sort = np.argsort(start_xs)
        start_xs = np.sort(start_xs)
        for i in range(len(bev_lanes)):
            bev_lanes_sorted[i] = bev_lanes[lane_sort[i]]

        for lane_id in range(len(start_xs)):  # 找是否存在车道线在车中间
            if np.abs(start_xs[lane_id]) < 4:
                positiontypes[0, lane_id] = 0
                middle_lane_exist = True
        if middle_lane_exist == True:  # 有中心线的情况
            middle_lane_id = lane_id
            for i in reversed(range(middle_lane_id)):
                positiontypes[0, i] = i-middle_lane_id
        else:  # 无中心线的情况
            len_left = 0  # 左边车道线数量
            for i in range(len(bev_lanes)):
                if start_xs[i] >= 0:
                    break
                len_left = len_left + 1
            if len_left > 0:
                left_len = len_left
                for i in reversed(range(len_left)):
                    if i-len_left == -1:  # 当要设定为左车道线时，判断横向距离过滤
                        if start_xs[i] <= -20:
                            left_len = left_len + 1
                    positiontypes[0, i] = i-left_len
            if len_left < len(bev_lanes_sorted):
                right_len = len_left
                for i in range(len_left, len(bev_lanes)):
                    if i-len_left == -1:  # 当要设定为右车道线时，判断横向距离过滤
                        if start_xs[i] >= 20:
                            right_len = right_len - 1
                    positiontypes[0, i] = i-right_len+1

        if 1 in positiontypes and -1 not in positiontypes and 0 not in positiontypes:  # 只检测出右侧车道时，估计左侧车道
            ego_left = bev_lanes_sorted[positiontypes.tolist()[0].index(1)]
            ego_left_fit = ego_left.fit
            endpoint = ego_left.endpoint
            startpoint = ego_left.startpoint
            pts = ego_left.pts
            pts_pre = []
            for pt in pts:
                pts_pre.append([pt[0]-20, pt[1]])
            bev_lane_pre = Bev_Lane(0, -1, 5, ego_left_fit, [endpoint[0]-20, endpoint[1]], [startpoint[0]-20, startpoint[1]], pts_pre)
            bev_lanes_sorted.append(bev_lane_pre)

        if -1 in positiontypes and 1 not in positiontypes and 0 not in positiontypes:  # 只检测出左侧车道时，估计右侧车道
            ego_right = bev_lanes_sorted[positiontypes.tolist()[0].index(-1)]
            ego_right_fit = ego_right.fit
            endpoint = ego_right.endpoint
            startpoint = ego_right.startpoint
            pts = ego_right.pts
            pts_pre = []
            for pt in pts:
                pts_pre.append([pt[0]+20, pt[1]])
            bev_lane_pre = Bev_Lane(0, 1, 5, ego_right_fit, [endpoint[0]+20, endpoint[1]], [startpoint[0]+20, startpoint[1]], pts_pre)
            bev_lanes_sorted.append(bev_lane_pre)

        for i in range(len(bev_lanes)):
            bev_lanes_sorted[i].position_type_ego = positiontypes[0, i]

        for i in reversed(range(len(bev_lanes_sorted))):
            if bev_lanes_sorted[i].position_type_ego not in [-2, -1, 0, 1, 2]:
                del bev_lanes_sorted[i]
    return bev_lanes_sorted


def line_positionnavtype_classfication(bev_lanes, nav_line):
    if nav_line != None:
        if len(bev_lanes) >= 1:
            start_xs = []
            for bev_lane in bev_lanes:
                start_x = bev_lane.pts[-3][0]  # bev车道线起点横坐标
                start_xs.append(start_x-nav_line.pts_x[-1])  # bev车道起点距本车中心线横向距离

            for i in range(len(bev_lanes)):
                if start_xs[i] >= 0 and start_xs[i] < 20:
                    bev_lanes[i].position_type_nav = 1
                elif start_xs[i] >= -20 and start_xs[i] < 0:
                    bev_lanes[i].position_type_nav = 2
                elif start_xs[i] >= -40 and start_xs[i] < -20:
                    bev_lanes[i].position_type_nav = 3
    return bev_lanes


def driving_line_fit(bev_lanes):
    ego_line = []
    nav_line = []
    for bev_lane in bev_lanes:
        if bev_lane.position_type_ego == -1 or bev_lane.position_type_ego == 1:
            for i in range(len(bev_lane.pts)):
                ego_line.append(bev_lane.pts[i])

        if bev_lane.position_type_nav == 1 or bev_lane.position_type_nav == 2:
            for i in range(len(bev_lane.pts)):
                nav_line.append(bev_lane.pts[i])



def line_classification(img, lines):  # 判断虚实线
    line_cls = [] # 车道线类型，实线：1，虚线：0
    for line in lines:
        colors = []
        for i in range(len(line)-1):
            if line[i][1] >= 700 or line[i][1] <= 400:
                continue
            pt_pre = [int((line[i][0] + line[i + 1][0]) / 2), int((line[i][1] + line[i + 1][1]) / 2)]
            colors.append([np.average(img[line[i][1], line[i][0], :])])
            colors.append([np.average(img[pt_pre[1], pt_pre[0], :])])
        if len(colors) >= 1:
            colors = np.array(colors)
            db = DBSCAN(eps=20, min_samples=2).fit(colors)
            k = np.max(db.labels_)
        else:
            k = -1
        if k == 0:
            line_cls.append(1)
        else:
            line_cls.append(0)
    return line_cls