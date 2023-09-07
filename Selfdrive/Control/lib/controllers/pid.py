import numpy as np
import copy


class PID:
    """Class of PID controller
    """
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ep = 0.0
        self.ei = 0.0
        self.ed = 0.0
        self.dt = 0.1

    def update_e(self, e):
        self.ed = e - self.ep
        self.ei += e
        self.ep = copy.deepcopy(e)

    def get_u(self):
        u = self.kp * self.ep + self.ki * self.ei + self.kd * self.ed
        if u > np.pi / 6:
            u = np.pi / 6
        if u < -np.pi / 6:
            u = -np.pi / 6
        return u

    def get_a(self):
        a = self.kp * self.ep + self.ki * self.ei + self.kd * self.ed
        return -a
