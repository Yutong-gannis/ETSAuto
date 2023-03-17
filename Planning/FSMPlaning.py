import time
from transitions import Machine

class FSMPlanner(object):
    states = ['Cruise', 'Follow', 'Passing']

    def __init__(self, name):
        self.name = name
        self.kittens_rescued = 0  # 记录决策数
        self.machine = Machine(model=self, states=FSMPlanner.states, initial='Cruise')  # 初始化状态机
        self.machine.add_transition(trigger='Cruise2Follow', source='Cruise', dest='Follow')  # 巡航到跟随
        self.machine.add_transition(trigger='Follow2Cruise', source='Follow', dest='Cruise')  # 跟随到巡航
        self.machine.add_transition(trigger='Follow2Passing', source='Follow', dest='Passing')  # 跟随到超车
        self.machine.add_transition(trigger='Passing2Follow', source='Passing', dest='Follow')  # 超车到跟随
        self.machine.add_transition(trigger='Passing2Cruise', source='Passing', dest='Cruise')  # 超车到巡航

    def update_journal(self):  # 更新记录
        self.kittens_rescued += 1


class PlanTrigger:
    def __init__(self):
        self.value = None
        self.obstacles = None
        self.lanes = None

        self.ego_lane_infor = 0  # [0：前方区域无车辆或障碍物 1：前车处于跟踪范围内]
        self.speed_infor = 0  # [0：前车纵向速度低于最低跟踪速度 1：前车纵向速度高于最高跟踪速度，]
        # self.near_lane_infor = [0, 0]  # [0：无左车道 1：有左车道，0：无右车道 1：有右车道]
        self.near_car_infor = [0, 0]  # [0：左侧车道无车辆行驶 1：左相邻车道有车辆行驶，0：右侧车道无车辆行驶 1：右侧车道有车辆行驶]
        self.t = 0  # 状态时间
        self.t_sheld = 1  # [0：不满足车道变换时间阈值 1：满足车道变换时间阈值]
        self.change_lane_state = 0  # [0：不在超车状态 1：驶入超车道 2：超车中 3：准备驶出超车道]

        self.state_now = 'Cruise'
        self.state_trigger = None


    def update_infor(self, obstacles, lanes):  # 更新基础条件
        self.obstacles = obstacles
        self.ego_lane_infor = 0
        self.near_car_infor = [0, 0]

        for lane in lanes:
            if lane.position_type == -1:
                detect_range = lane.startpoint[1]
            # if lane.position_type == -2:
            #     self.near_lane_infor[0] == 1
            # elif lane.position_type == 2:
            #     self.near_lane_infor[1] == 1

        for obstacle in self.obstacles:  # 更新周围汽车信息
            if obstacle.lane == 0 and obstacle.position[1] + obstacle.position[3] >= detect_range:
                self.ego_lane_infor = 1
                if self.state_now == 'Follow':
                    if 13 < time.time() - self.t < 45:
                        self.t_sheld = 1
                    elif time.time() - self.t >= 45:
                        self.t_sheld = 0
                if obstacle.speed[1] <= -1:
                    self.speed_infor = 1
                elif obstacle.speed[1] >= -1:
                    self.speed_infor = 0

            elif obstacle.lane == -1 and obstacle.position[1] + obstacle.position[3] >= 420:
                self.near_car_infor[0] = 1

            elif obstacle.lane == 1 and obstacle.position[1] + obstacle.position[3] >= 420:
                self.near_car_infor[1] = 1

            elif obstacle.lane == 1 and self.change_lane_state == 2 and obstacle.position[1] + obstacle.position[3] >= detect_range:
                self.near_car_infor[1] = 1

        if self.state_now == 'Passing' and self.change_lane_state == 2 and self.near_car_infor[1] == 0:
            self.t = time.time()
            self.change_lane_state = 3

        if self.state_now == 'Passing' and self.change_lane_state == 3 and self.near_car_infor[1] == 1:
            self.change_lane_state = 2

    def update_trigger(self, state_now, obstacles, lane, info):
        self.update_infor(obstacles, lane)

        self.state_now = state_now
        if self.state_now == 'Cruise' and self.ego_lane_infor == 1:
            self.state_trigger = 'Cruise2Follow'
            self.t = time.time()
        elif self.state_now == 'Follow' and self.ego_lane_infor == 0:
            self.state_trigger = 'Follow2Cruise'
            self.t = 0
        elif self.state_now == 'Follow' and info.roads_type == 1 and self.near_car_infor[0] == 0 and self.t_sheld == 1:
            self.state_trigger = 'Follow2Passing'
            self.change_lane_state = 1
        elif self.state_now == 'Passing' and self.ego_lane_infor == 1 and self.t_sheld == 0:
            self.state_trigger = 'Passing2Follow'
            self.t = time.time()
        elif self.state_now == 'Passing' and self.ego_lane_infor == 0 and self.t_sheld == 0:
            self.state_trigger = 'Passing2Cruise'
            self.t = 0
        else:
            self.state_trigger = None
        return self.state_trigger
