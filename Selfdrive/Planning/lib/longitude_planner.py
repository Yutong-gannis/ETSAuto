import numpy as np
import scipy

class LongitudePlanner:
    def __init__(self):
        self.speed_list = []
        self.plan_length = 20
        self.smooth_window = 10

    def get_arc_curve(self, pts):
        start = np.array(pts[0])
        end = np.array(pts[len(pts) - 1])
        l_arc = np.sqrt(np.sum(np.power(end - start, 2)))
        a = l_arc
        b = np.sqrt(np.sum(np.power(pts - start, 2), axis=1))
        c = np.sqrt(np.sum(np.power(pts - end, 2), axis=1))
        dist = np.sqrt((a + b + c) * (a + b - c) * (a + c - b) * (b + c - a)) / (2 * a)
        h = dist.max()
        r = ((a * a) / 4 + h * h) / (2 * h)
        return r

    def update(self, trajectory, speed_now):
        arc = self.get_arc_curve(trajectory)
        speed_plan = 20 * arc ** 0.3168 / 3.6
        self.speed_list.append(speed_plan)
        if len(self.speed_list) > self.plan_length:
            self.speed_list.pop(0)
        if len(self.speed_list) >= self.smooth_window:
            speed_limit_np = self.speed_list
            speed_limit_smooth = scipy.signal.savgol_filter(speed_limit_np, self.smooth_window, 1, axis=0)
            speed_plan =  speed_limit_smooth[-1]
        return speed_plan / 2 + speed_now / 2