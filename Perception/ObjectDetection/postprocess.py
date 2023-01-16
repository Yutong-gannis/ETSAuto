import numpy as np
import torch

class Traffic_Light():  # 对红绿灯信息进行封装
    def __init__(self, tlcolor_type, distance):
        self.tlcolor = tlcolor_type  # 定义红绿灯状态
        ###红绿灯状态###
        # 绿：0
        # 红：1
        # 黄：2
        ##############
        self.distance = distance

class Obstacle:  # 对障碍物信息进行封装
    def __init__(self, id, position, cls, speed, track):
        self.id = id
        self.position = position  # 左上角位置
        self.cls = cls  # 类别：0：小轿车 1：巴士、卡车
        self.speed = speed  # 速度向量[speed_x, speed_y]
        self.track = track  # 保存前五帧的轨迹
        self.lane = None
        #####所属车道信息#####
        # 左侧车道：-1
        # 同一车道：0
        # 右侧车道：1
        # 未知：None
        ####################
        self.cipv = None # 是否为前方最邻近车辆 是：True


def postprocess(dets, CAM, CAM_BL, tracker, tracks, infer_time):  # 目标追踪后处理主函数
    objs, traffic_lights_list = position(dets, CAM, CAM_BL)
    if len(objs) >= 1:
        objs = np.array(objs)
        targets = tracker.update(torch.tensor(objs), [600, 800], (600, 800))
        tracks, obstacles = update_tracks(tracks, targets, infer_time)
    else:
        obstacles = None
        objs = None
    if len(traffic_lights_list):
        traffic_light = tl_filter(traffic_lights_list)
    else:
        traffic_light = None
    return obstacles, objs, tracks, traffic_light


def position(dets, CAM, CAM_BL):  # 定位
    objs = []
    traffic_lights_list = []
    for det in dets:
        box = np.copy(det)
        if box[5] == 0 or box[5] == 1 or box[5] == 2 or box[5] == 3 or box[5] == 4 and box[4] >= 0.5:
            x0 = box[0]  # 左边界
            x1 = box[2]  # 右边界
            y1 = box[3]  # 下底
            if (x0+x1)/2 > 0 and (x0+x1)/2 < 180 and y1 <= 240: # 处理左后视镜
                position = CAM_BL.pixel_to_world([(x0 + x1) / 2, 240 - y1])[0]  # 相对距离
                if box[5] == 2:
                    box[0] = 392 + int((position[0, 0]))
                    box[1] = 470 + int(position[0, 1])
                    box[2] = 400 + int((position[0, 0]))
                    box[3] = 490 + int(position[0, 1])
                elif box[5] == 3 or box[5] == 4:
                    box[0] = 392 + int((position[0, 0]))
                    box[1] = 470 + int(position[0, 1])
                    box[2] = 400 + int((position[0, 0]))
                    box[3] = 530 + int(position[0, 1])
            elif (x0+x1)/2 < 1280 and (x0+x1)/2 > 1100 and y1 <= 240: # 处理左后视镜
                position = CAM_BL.pixel_to_world([(x0 + x1) / 2, 240 - y1])[0]  # 相对距离
                if box[5] == 2:
                    box[0] = 400 + int((position[0, 0]))
                    box[1] = 470 + int(position[0, 1])
                    box[2] = 408 + int((position[0, 0]))
                    box[3] = 490 + int(position[0, 1])
                elif box[5] == 3 or box[5] == 4:
                    box[0] = 400 + int((position[0, 0]))
                    box[1] = 470 + int(position[0, 1])
                    box[2] = 408 + int((position[0, 0]))
                    box[3] = 530 + int(position[0, 1])
            else:  # 处理前视摄像头
                if box[5] == 2:
                    position = CAM.pixel_to_world([(x0+x1)/2, 720 - y1])[0]
                    box[0] = 396 + int((position[0, 0] / 5))
                    box[1] = 450 - int(position[0, 1] / 1.8)
                    box[2] = 404 + int((position[0, 0] / 5))
                    box[3] = 470 - int(position[0, 1] / 1.8)
                elif box[5] == 3 or box[5] == 4:
                    position = CAM.pixel_to_world([(x0 + x1) / 2, 720 - y1])[0]
                    box[0] = 396 + int((position[0, 0] / 5))
                    box[1] = 410 - int(position[0, 1] / 1.8)
                    box[2] = 404 + int((position[0, 0] / 5))
                    box[3] = 470 - int(position[0, 1] / 1.8)
                if x1 >= 1270 and x1-x0 >= 50:  # 对近处的物体进行横向距离补偿
                    box[0] = 416
                    box[2] = 424
                elif x0 <= 10 and x1-x0 >= 50:  # 对近处的物体进行横向距离补偿
                    box[0] = 376
                    box[2] = 384
            if box[3] <= 250 or np.abs((box[0] + box[2]) / 2 - 400) >= 50:
                continue
            objs.append(box)

        elif box[5] == 5 or box[5] == 6 or box[5] == 7 and box[4] >= 0.95:  # 红绿灯
            s = (box[2] - box[0]) * (box[3] - box[1])  # 面积
            distance = scale_depth(s, f=0.07) * 5
            traffic_lights_list.append([box[5]-5, distance])
    return objs, traffic_lights_list


def tl_filter(tl_list):  # 对红绿灯信息进行整合
    tl_list = np.array(tl_list)
    tl_color = tl_list[:, 0].astype(np.int8)
    tl_distance = tl_list[:, 1]
    tl_color_filted = np.argmax(np.bincount(tl_color))
    tl_distance_filted = np.average(tl_distance)
    traffic_light = Traffic_Light(tl_color_filted, tl_distance_filted)
    return traffic_light

def update_tracks(tracks, targets, infer_time):
    ids = []
    new_tracks = []
    obstacles = []
    for t in targets:
        tlwh = t.tlwh
        tid = t.track_id
        ids.append(int(tid))
        if len(tracks) == 0:
            tracks.append([int(tid), [tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score]])
        else:
            for i in range(len(tracks)):
                if tid == tracks[i][0]:
                    tracks[i].append([tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score])
                    break
                if i == len(tracks) - 1:
                    tracks.append([int(tid), [tlwh[0], tlwh[1], tlwh[2], tlwh[3], t.score]])

    for tid in ids:  # 当前出现的存入新的轨迹中
        for track in tracks:
            if tid == track[0]:
                if len(track) >= 6:
                    track = [track[0]] + track[-5:]
                new_tracks.append(track)

    if len(new_tracks) >= 0:
        for track in new_tracks:
            pos_now = track[-1]
            speed_x, speed_y = speed_estimation(track, infer_time)

            if pos_now[3] >= 50:
                cls = 1
                pos_now[3] = 60
            else:
                cls = 0
                pos_now[3] = 20
            obstacle = Obstacle(track[0], pos_now, cls, [speed_x, speed_y], track[1:]) # 障碍物封装
            obstacles.append(obstacle)
    else:
        obstacles = None
    return new_tracks, obstacles


def scale_depth(s, f=0.07):  # 几何方法测红绿灯距离 f: 焦距
    common_size = 0.75
    proportion = np.sqrt(common_size / (s*0.0002**2))
    distance = (proportion+1)*f
    return distance


def speed_estimation(track, infer_time):  # 计算速度差
    if len(track) >= 4:
        pts = np.array(track[1:])
        fit = np.polyfit(pts[:, 1], pts[:, 0], 1)
        speed_y = (track[-1][1] - track[-3][1]) / (infer_time * 2)  # 计算周围汽车速度
        speed_x = speed_y * fit[0]
    elif len(track) == 3:
        speed_y = (track[-1][1] - track[-2][1]) / infer_time
        speed_x = (track[-1][0] - track[-2][0]) / infer_time
    else:
        speed_x = 0
        speed_y = 0
    return speed_x, speed_y
