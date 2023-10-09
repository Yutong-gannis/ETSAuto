import os
import sys
import math
import numpy as np
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.insert(0, project_path)


class ObjectRegister:
    def __init__(self):
        self.distances = [50, 50, 50]  # 轨迹及左右最近的车
        self.leader_car_list = []
        self.speed = 0
        self.dtheta = 0
        self.time_last = 0.0
        self.theta_last = 0
        self.dt = 0.0
        self.dX = 0.0
        self.dY = 0.0
    
    def update_condition(self, time_now, condition_dict):
        self.time_last = time_now
        if condition_dict is not None:
            self.dtheta = self.theta_last - condition_dict['theta']
            self.theta_last = condition_dict['theta']
            self.speed = condition_dict['speed']
            self.dX = self.speed * np.cos(self.dtheta) * self.dt
            self.dY = self.speed * np.sin(self.dtheta) * self.dt


    def update(self, dets_dict, trajectory, condition_dict, lane_width):
        on_trajectory_obj = []
        left_trajectory_obj = []
        right_trajectory_obj = []
        if condition_dict is not None:
            self.speed = condition_dict['speed']
        if dets_dict is not None:
            objects = dets_dict['objects']
            if trajectory is not None and objects is not None:
                for i in range(len(objects)):
                    if len(trajectory) > int(objects[i, 0] * 2):
                        trajectory_in_distance = trajectory[int(objects[i, 0] * 2), 1]
                    else:
                        continue
                    if abs(trajectory_in_distance - objects[i, 1]) <= lane_width/2:
                        on_trajectory_obj.append(objects[i])
                    elif trajectory_in_distance - objects[i, 1] >= lane_width/2 and trajectory_in_distance - objects[i, 1] <= lane_width * 1.5:
                        left_trajectory_obj.append(objects[i])
                    elif trajectory_in_distance - objects[i, 1] >= -lane_width * 1.5 and trajectory_in_distance - objects[i, 1] <= -lane_width/2:
                        right_trajectory_obj.append(objects[i])
            elif trajectory is None and objects is not None:
                for i in range(len(objects)):
                    if abs(objects[i, 1]) <= lane_width/2:
                        on_trajectory_obj.append(objects[i])
                    elif objects[i, 1] >= -lane_width * 1.5 and objects[i, 1] <= -lane_width/2:
                        left_trajectory_obj.append(objects[i])
                    elif objects[i, 1] >= lane_width/2 and objects[i, 1] <= lane_width * 1.5:
                        right_trajectory_obj.append(objects[i])

        on_trajectory_obj = np.array(on_trajectory_obj)
        left_trajectory_obj = np.array(left_trajectory_obj)
        right_trajectory_obj = np.array(right_trajectory_obj)
        
        if len(on_trajectory_obj):
            front_nearest_obj = on_trajectory_obj[np.argmin(on_trajectory_obj[:, 0])]
            self.leader_car_list.append(front_nearest_obj)
            if len(self.leader_car_list) > 10:
                self.leader_car_list.pop(0)
            self.distances[1] = front_nearest_obj[0]
        else:
            self.leader_car_list = []
            self.distances[1] = 50

        if len(left_trajectory_obj):
            left_nearest_obj = left_trajectory_obj[np.argmin(left_trajectory_obj[:, 0])]
            self.distances[0] = left_nearest_obj[0]
        else:
            self.distances[0] = 50

        if len(right_trajectory_obj):
            right_nearest_obj = right_trajectory_obj[np.argmin(right_trajectory_obj[:, 0])]
            self.distances[2] = right_nearest_obj[0]
        else:
            self.distances[2] = 50
        
        fcw, aeb = False, False
        leader_position, leader_speed = None, None
        if len(self.leader_car_list) >= 3:
            leader_car_np = np.array(self.leader_car_list)
            dspeed = np.sign(leader_car_np[-1, 0] - leader_car_np[0, 0]) * np.linalg.norm(leader_car_np[-1, :] - leader_car_np[0, :]) / ((len(leader_car_np) - 1) * 0.05)
            # print(dspeed)
            self.leader_car_speed = self.speed + dspeed
            leader_position = leader_car_np[-1, :]
            leader_speed = self.leader_car_speed
            fcw, aeb = self.TTC(leader_car_np[-1, :], self.leader_car_speed)

        self.publish_fcw(fcw)
        return leader_position, leader_speed
        
    def TTC(self, leader_position, leader_speed):
        fcw, aeb = False, False
        ttc = np.linalg.norm(leader_position) / (self.speed - leader_speed)  # TTC碰撞时间
        if ttc > 0 and ttc <= 2.7 and self.speed > 1 and leader_speed > 1:
            fcw = True
        if ttc > 0 and ttc <= 1 and self.speed > 5 and leader_speed > 5:
            aeb = True
        return fcw, aeb
    
    def publish_fcw(self, fcw):
        fcw_dict_pub = SharedMemoryDict(name='fcw', size=1024)
        fcw_dict_pub['fcw'] = fcw
