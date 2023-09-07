import numpy as np
import scipy
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from transform import trans_rotate, trans_translate


class PlanRegister:
    def __init__(self):
        self.history_position = None
        self.history_length = 10
        self.smooth_window = 5

    def update(self, plan_position):
        if plan_position is None:
            return None
        if self.history_position is None:
            self.history_position = np.expand_dims(plan_position, axis=0)
            # history_position_smooth = scipy.signal.savgol_filter(self.history_position, 3, 1, axis=1)  # 轨迹平滑
            history_position_smooth = self.history_position
        else:
            if len(self.history_position) >= self.history_length:
                self.history_position = np.delete(self.history_position, 0, axis=0)
            self.history_position = np.concatenate((self.history_position, np.expand_dims(plan_position, axis=0)), axis=0)
            # history_position_smooth = scipy.signal.savgol_filter(self.history_position, 3, 1, axis=1)  # 轨迹平滑
            history_position_smooth = self.history_position
            if len(self.history_position) >= self.smooth_window:
                history_position_smooth = scipy.signal.savgol_filter(history_position_smooth, self.smooth_window, 1, axis=0)  # 时间序列平滑
        plan_position_smooth = history_position_smooth[-1]
        return plan_position_smooth
