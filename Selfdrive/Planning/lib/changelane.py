import os
import sys
import math
import numpy as np
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.insert(0, project_path)
from lib.optimizers.bazier_optimizer import point_on_bezier_curve
from lib.transform import update_trajectory
from lib.planregister import PlanRegister


defaut_change_distance = 50


class ChangeLane_Helper:
    """Class to help change lane
    """
    def __init__(self):
        self.lane_change_state = 0
        self.trajectory_change = None
        self.change_distance = defaut_change_distance
        self.hist_lane_l = None
    
    def plan_change_start(self, trajectory, desire, lane_width):
        """Using bazier curve to plan trajectory in the first stage of change lane

        :param trajectory: Trajectory for going ahead
        :type trajectory: np.array
        :param desire: Desire of user
        :type desire: str
        :param lane_width: Width of lane
        :type lane_width: float
        """
        trajectory_theta = np.arctan((trajectory[3, 1] - trajectory[0, 1]) / (trajectory[3, 0] - trajectory[0, 0]))
        if desire == 'changelaneleft':
            line_target = trajectory - [[-lane_width * np.sin(trajectory_theta), lane_width * np.cos(trajectory_theta)]]
        else:
            line_target = trajectory + [[-lane_width * np.sin(trajectory_theta), lane_width * np.cos(trajectory_theta)]]

        P0 = [0, 0]
        P1 = trajectory[5, :]
        if self.change_distance * 2 + 10 >= len(line_target):
            P2 = line_target[-5, :]
            P3 = line_target[-1. :]
        else:
            P2 = line_target[5 + self.change_distance * 2, :]
            P3 = line_target[5 + self.change_distance * 2 + 5, :]

        a = 2 / 5
        Ay = P1[1]
        Ax = P1[0] + a * (P2[0] - P0[0])
        A = np.array([Ax, Ay])

        b = 2 / 5
        By = P2[1]
        Bx = P2[0] - b * (P3[0] - P1[0])
        B = np.array([Bx, By])

        ts = np.linspace(0, 1, self.change_distance * 2 + 1)

        Q = np.zeros((self.change_distance * 2 + 1, 2))
        for i, t in enumerate(ts):
            Q[i, :] = point_on_bezier_curve([P1, A, B, P2], t)
        self.trajectory_change = np.concatenate((trajectory[:5, :], Q), axis=0)

        pts_x = np.linspace(0, math.ceil(self.trajectory_change[-1, 0]), math.ceil(self.trajectory_change[-1, 0]) * 2 + 1)
        fit_m = np.polyfit(self.trajectory_change[:, 0], self.trajectory_change[:, 1], 3)
        pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
        self.trajectory_change = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1))), axis=1)

    def plan_change_subject(self, trajectory, line_l, lane_width, desire, condition_dict):
        """Plan trajectory in the second stage of change lane

        :param trajectory: History trajectory
        :type trajectory: np.array
        :param line_l: Left line of ego lane
        :type line_l: np.array
        :param lane_width: Width pof lane
        :type lane_width: float
        :param desire: Desire of user
        :type desire: str
        :param condition_dict: Dictionary of truck condition
        :type condition_dict: dict
        """
        self.trajectory_change = update_trajectory(self.trajectory_change, condition_dict)
        
        trajectory_theta = np.arctan((trajectory[3, 1] - trajectory[0, 1]) / (trajectory[3, 0] - trajectory[0, 0]))

        if desire == 'changelaneleft':
            line_target = trajectory - [[-lane_width * np.sin(trajectory_theta), lane_width * np.cos(trajectory_theta)]]
        else:
            line_target = trajectory + [[-lane_width * np.sin(trajectory_theta), lane_width * np.cos(trajectory_theta)]]
        
        trajectory_total = np.vstack((self.trajectory_change[:5, :], line_target[self.change_distance * 2 + 1:, :]))

        pts_x = np.linspace(0, math.ceil(self.trajectory_change[-1, 0]), math.ceil(self.trajectory_change[-1, 0]) * 2 + 1)
        
        fit_m = np.polyfit(trajectory_total[:, 0], trajectory_total[:, 1], 3)
        pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
        self.trajectory_change = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1))), axis=1)
        
        self.check_state(line_l, condition_dict)
    

    def check_state(self, line_l, condition_dict):
        """Check state if truck can turn to third stage

        :param line_l: left lane of ego lane
        :type line_l: np.array
        :param condition_dict: Dictionary of truck condition
        :type condition_dict: dict
        """
        if line_l is not None and len(line_l) >= 30:
            if self.hist_lane_l is None:
                self.hist_lane_l = line_l
            elif self.hist_lane_l is not None and abs(np.average(self.hist_lane_l[:20, 1]) - np.average(line_l[:20, 1])) >= 2:
                self.lane_change_state = 2
                self.hist_lane_l = None
            else:
                self.hist_lane_l = line_l
            if self.hist_lane_l is not None:
                self.hist_lane_l = update_trajectory(self.hist_lane_l, condition_dict)

    
    def plan_change_end(self, trajectory, line_l, condition_dict):
        self.trajectory_change = update_trajectory(self.trajectory_change, condition_dict)
        if line_l is not None and len(line_l) >= 30:
            if self.hist_lane_l is not None and abs(np.average(self.hist_lane_l[:20, 1]) - np.average(line_l[:20, 1])) < 2:
                trajectory_total = np.vstack(([[-3, 0], [-1, 0]], trajectory[-10:, :]))
                self.hist_lane_l = line_l
            else:
                trajectory_total = self.trajectory_change
        else:
            trajectory_total = self.trajectory_change

        pts_x = np.linspace(0, math.ceil(self.trajectory_change[-1, 0]), math.ceil(self.trajectory_change[-1, 0]) * 2 + 1)
        
        fit_m = np.polyfit(trajectory_total[:, 0], trajectory_total[:, 1], 3)
        pts_m_y = fit_m[0] * pts_x ** 3 + fit_m[1] * pts_x ** 2 + fit_m[2] * pts_x + fit_m[3]
        self.trajectory_change = np.concatenate((pts_x.reshape((-1, 1)), pts_m_y.reshape((-1, 1))), axis=1)
        if abs(trajectory[5, 1]) <= 0.5:
            self.trajectory_change = None
            self.lane_change_state = 3
            
    def plan_change_before(self, condition_dict):
        self.trajectory_change = update_trajectory(self.trajectory_change, condition_dict)

    def update(self, trajectory, line_l, plan_register, lane_width, desire, condition_dict):
        if desire in ['changelaneleft', 'changelaneright']:
            if trajectory is not None:
                if self.trajectory_change is None and self.lane_change_state == 0:  # 辅助变道规划
                    plan_register = PlanRegister()
                    self.lane_change_state = 1
                    self.plan_change_start(trajectory, desire, lane_width)

                if self.trajectory_change is not None and self.lane_change_state == 1:
                    self.plan_change_subject(trajectory, line_l, lane_width, desire, condition_dict)
                
                if self.trajectory_change is not None and self.lane_change_state == 2:
                    self.plan_change_end(trajectory, line_l, condition_dict)
                    self.publish()
                
            else:
                self.plan_change_before(condition_dict)

        elif desire not in ['changelaneleft', 'changelaneright']:
            self.trajectory_change = None
            self.lane_change_state = 0
            self.publish()
        
        if self.trajectory_change is not None:
            return self.trajectory_change, plan_register
        else:
            return trajectory, plan_register

    def publish(self):
        states_dict_pub = SharedMemoryDict(name='states', size=1024)
        states_dict_pub['lane_change_state'] = self.lane_change_state
