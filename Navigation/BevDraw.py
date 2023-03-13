import cv2
import numpy as np


def draw_bev(nav_line, obstacles, traffic_light, bev_lanes, stop_line):  # 处理nav及bev可视化的主函数
    bevmap = np.ones((600, 800)) * 75
    bevmap = draw_ruler(bevmap)
    bevmap = draw_nav_line(bevmap, nav_line.pts)
    bevmap = cv2.cvtColor(np.uint8(bevmap), cv2.COLOR_GRAY2BGR)
    bevmap = display_bev_lanes(bevmap, bev_lanes, stop_line, width=2)
    bevmap = display_objects(bevmap, obstacles, traffic_light)
    return bevmap

def draw_ruler(bevmap):
    bevmap = cv2.line(bevmap, (500, 470), (500, 200), [255, 255, 255], thickness=1, lineType=4)
    for x in range(200, 470, 30):
        bevmap = cv2.line(bevmap, (495, x), (500, x), [255, 255, 255], thickness=1, lineType=4)
    return bevmap


def draw_nav_line(bevmap, nav_pts):  # 将导航线绘制在BEV
    if len(nav_pts):
        for nav_pt in nav_pts:
            cv2.circle(bevmap, [int(nav_pt[0]), int(nav_pt[1])], radius=1, color=(0, 0, 0), thickness=-1)
    return bevmap


def display_objects(bevmap, obstacles, traffic_light):
    bevmap = cv2.rectangle(bevmap, (394, 470), (406, 530), [255, 255, 255], thickness=-1, lineType=4) # 卡车
    if obstacles is not None:
        for obstacle in obstacles:
            pos = obstacle.position
            x_middle = int(pos[0]+pos[2]/2)
            y_middle = int(pos[1]+pos[3]/2)
            speed = obstacle.speed
            if np.abs(x_middle-400) >= 50 or y_middle <= 250:
                continue
            obstacle_color = [int(pos[4]*125), int(pos[4]*125), int(pos[4]*125)]
            if obstacle.lane == 0:
                obstacle_color = [0, 0, 0]
                cv2.putText(bevmap, '{:^.1f}'.format(obstacle.speed[1]), (int(pos[0]), int(pos[1]-1)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 0), thickness=1)
                cv2.putText(bevmap, '{:^.1f}'.format(470 - (pos[1] + pos[3])), (int(pos[0]), int(pos[1] + pos[3] + 10)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4, color=(0, 0, 0), thickness=1)
            elif obstacle.lane == -1:
                obstacle_color = [0, 50, 0]
            elif obstacle.lane == 1:
                obstacle_color = [0, 0, 50]
            bevmap = cv2.rectangle(bevmap, (int(pos[0]), int(pos[1])), (int(pos[0]+8), int(pos[1]+pos[3])), obstacle_color, thickness=-1, lineType=4)
            bevmap = cv2.line(bevmap, (int(x_middle), int(y_middle)), (int(x_middle+speed[0]*0.4), int(y_middle+speed[1]*0.4)), [150, 150, 150], thickness=2, lineType=4)  # 卡车宽度变化因为识别为小汽车

    if traffic_light is not None:  # 绘制红绿灯
        distance = traffic_light.distance
        tlcolor = traffic_light.tlcolor
        if tlcolor == 0:  # 绿灯
            bevmap = cv2.circle(bevmap, (420, int(470 - distance)), 5, (0, 125, 0), -1)
        elif tlcolor == 1:  # 红灯
            bevmap = cv2.circle(bevmap, (420, int(470 - distance)), 5, (0, 0, 125), -1)
        elif tlcolor == 2:  # 黄灯
            bevmap = cv2.circle(bevmap, (420, int(470 - distance)), 5, (0, 125, 125), -1)

    return bevmap


def display_bev_lanes(bevmap, bev_lanes, stop_line, width=2):
    if len(bev_lanes) >= 1:
        for bev_lane in bev_lanes:
            bev_lane_pts = bev_lane.pts
            if bev_lane.position_type in [-1.0, 0.0, 1.0]:  # 临近车道线设为重点
                width = 2
            else:
                width = 1
            for i in range(1, len(bev_lane_pts)):
                cv2.line(bevmap, bev_lane_pts[i - 1], bev_lane_pts[i], [200, 200, 200], thickness=width)
            '''
            if bev_lane.cls == 1:  # 画实线
                for i in range(1, len(bev_lane_pts)):
                    cv2.line(bevmap, bev_lane_pts[i - 1], bev_lane_pts[i], [200, 200, 200], thickness=width)
            else:  # 画虚线
                #time_now = int(time.time()*100) % 10  # 移动效果，没鸟用
                for i in range(1, len(bev_lane_pts)):
                    if i%5 == 1 or i%5 == 2:
                        continue
                    cv2.line(bevmap, bev_lane_pts[i - 1], bev_lane_pts[i], [200, 200, 200], thickness=width)
            '''

    if stop_line is not None:
        stop_line_y = stop_line.center_y
        length = stop_line.length
        cv2.line(bevmap, [int(400-length/2), int(470-stop_line_y)], [int(400+length/2), int(470-stop_line_y)], [255, 255, 255], thickness=2)

    return bevmap

def print_info(bevmap, refer_time, truck, speed_limit, state, weather, info):
    cv2.putText(bevmap, 'Infer time: {:.4f}s'.format(refer_time), (10, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(200, 200, 200), thickness=1)
    cv2.putText(bevmap, "Steer angle: {:.2f}".format((truck.ang - 0.5) * 180), (10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    cv2.putText(bevmap, "speed limit: {}".format(speed_limit), (10, 45), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    cv2.putText(bevmap, "speed: {}".format(truck.speed), (10, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    cv2.putText(bevmap, "power: {:.1f}".format((1-truck.acc)*10), (10, 75), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    cv2.putText(bevmap, "ld: {:.4f}".format(truck.ld), (10, 120), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    cv2.putText(bevmap, "lf: {:.4f}".format(truck.lf), (10, 135), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)
    if not info.activeAP:
        cv2.putText(bevmap, "AP: {}".format(info.activeAP), (10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1)
        if info.AP_exit_reason == 1:
            cv2.putText(bevmap, "AP exited, please take control immediately!!!", (10, 332), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 255), thickness=1)
    elif info.roads_type == 0:
        cv2.putText(bevmap, "AP: road", (10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1)
    elif info.roads_type == 1:
        cv2.putText(bevmap, "AP: highway", (10, 150), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1)        
            

    if state is not None:
        cv2.putText(bevmap, state, (10, 90), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5, color=(255, 255, 255), thickness=1)

    cv2.putText(bevmap, "weather: {}".format(weather), (10, 105), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255, 255, 255), thickness=1)

    return bevmap