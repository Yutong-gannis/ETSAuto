import cv2
import numpy as np

class Nav_Line:  # 封装导航线信息
    def __init__(self, fit, pts, pts_x, pts_y):
        self.fit = fit
        self.pts = pts
        self.pts_x = pts_x
        self.pts_y = pts_y

def nav_process(nav):
    bev_nav = nav2bev(nav)
    nav_line = get_nav_line(bev_nav)
    curve = np.abs(nav_line.fit[0])
    if curve <= 0.0005:
        curve_speed_limit = 80
    elif curve > 0.0005 and curve <= 0.001:
        curve_speed_limit = 75
    elif curve > 0.001 and curve <= 0.002:
        curve_speed_limit = 65
    elif curve > 0.002 and curve <= 0.004:
        curve_speed_limit = 55
    elif curve > 0.004 and curve <= 0.005:
        curve_speed_limit = 40
    elif curve > 0.005:
        curve_speed_limit = 20
    elif curve == 0:
        curve_speed_limit = 30
    return nav_line, curve_speed_limit

def nav2bev(map):  # 将导航地图转化为bev画布
    pts_src = np.float32([
        [  0,   0], [200,   0],
        [  0, 130], [200, 130], ])

    pts_dst = np.float32([
        [  0,   0], [800,   0],
        [300, 600], [500, 600], ])
    h = cv2.getPerspectiveTransform(pts_src, pts_dst)
    map = cv2.warpPerspective(map, h, (800, 600))
    map = filter_out_red(map)
    return map

def filter_out_red(img):
    img_blue = img[:, :, 0]
    _, mask_blue = cv2.threshold(img_blue, 180, 125, cv2.THRESH_BINARY)
    img_red = img[:, :, 2]
    _, mask_red = cv2.threshold(img_red, 100, 125, cv2.THRESH_BINARY_INV)
    mask = mask_blue + mask_red
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.dilate(mask, kernel, iterations=6)
    mask = cv2.Canny(mask, 50, 150)
    _, mask = cv2.threshold(mask, 180, 180, cv2.THRESH_BINARY)
    mask = mask+75
    return mask

def get_nav_line(img):
    middle_pts = []
    x = 400
    for i in range(540, 340, -5):
        for j in range(x, 0, -1):
            if img[i, j] == 255:
                break
        for k in range(x, 800, 1):
            if img[i, k] == 255:
                break
        if j == 0 or j == 799 or k == 0 or k == 799:
            break
        x = int((j+k)/2)
        middle_pts.append([x, i])
    middle_pts = np.array(middle_pts) # 路线中心线
    if len(middle_pts):
        fit = np.polyfit(np.array([i[1] for i in middle_pts]), np.array([i[0] for i in middle_pts]), 2)
        pts_y = np.linspace(300, 470, 50)
        pts_x = fit[0] * pts_y ** 2 + fit[1] * pts_y + fit[2]
    else:
        pts_x = [400]
        pts_y = [470]
        fit = [0, 0, 0]
    nav_pts = []
    if len(pts_y):
        for pt_x, pt_y in zip(pts_x, pts_y):
            nav_pt = [pt_x, pt_y]
            nav_pts.append(nav_pt)
    nav_line = Nav_Line(fit, nav_pts, pts_x, pts_y)
    return nav_line