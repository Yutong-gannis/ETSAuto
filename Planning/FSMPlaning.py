from transitions import Machine

class FSMPlanner(object):
    states = ['Cruise', 'Follow', 'Passing', 'ActiveCollisionAvoidance']  # 目前不允许切换到超车状态

    def __init__(self, name):
        self.name = name
        self.kittens_rescued = 0  # 记录决策数
        self.machine = Machine(model=self, states=FSMPlanner.states, initial='Cruise')  # 初始化状态机
        self.machine.add_transition(trigger='Cruise2Follow', source='Cruise', dest='Follow')  # 定速巡航到跟随
        self.machine.add_transition(trigger='Follow2Cruise', source='Follow', dest='Cruise')  # 跟随到自适应巡航
        # self.machine.add_transition(trigger='Follow2Passing', source='Follow', dest='Passing')  # 跟随到超车
        # self.machine.add_transition(trigger='Passing2Follow', source='Passing', dest='Follow')  # 超车到跟随
        # self.machine.add_transition(trigger='Passing2Cruise', source='Passing', dest='Cruise')  # 超车到巡航
        self.machine.add_transition(trigger='Cruise2ACA', source='Cruise', dest='ActiveCollisionAvoidance')  # 定速巡航到主动避障
        self.machine.add_transition(trigger='Follow2ACA', source='Follow', dest='ActiveCollisionAvoidance')  # 跟随到主动避障
        self.machine.add_transition(trigger='ACA2Follow', source='ActiveCollisionAvoidance', dest='Follow')  # 主动避障到跟随
        # self.machine.add_transition(trigger='Passing2ACA', source='Passing', dest='ActiveCollisionAvoidance')  # 超车到主动避撞
        self.machine.add_transition(trigger='ACA2Cruise', source='ActiveCollisionAvoidance', dest='Cruise')  # 主动避障到定速巡航

    def update_journal(self):  # 更新记录
        self.kittens_rescued += 1


class PlanTrigger:
    def __init__(self):
        self.value = None
        self.obstacles = None
        self.lanes = None

        self.ego_lane_infor = 0  # 0：前方区域无车辆或障碍物 1：前车处于跟踪范围内 2：前车处于避障范围内]
        self.speed_infor = 0  # 0：前车纵向速度低于最低跟踪速度 1：前车纵向速度高于最高跟踪速度，]
        self.near_lane_infor = [0, 0]  # [0：有左车道 1：无左车道，0：有右车道 1：无右车道]
        self.near_car_infor = [0, 0]  # [0：左侧车道无车辆行驶 1：左相邻车道有车辆行驶，0：右侧车道无车辆行驶 1：右侧车道有车辆行驶]
        self.t_sheld = 1  # 0：不满足车道变换时间阈值 1：满足车道变换时间阈值

        self.state_now = 'Cruise'
        self.state_trigger = None

    def update_infor(self, obstacles, lanes):  # 更新基础条件
        self.obstacles = obstacles
        self.ego_lane_infor = 0

        for lane in lanes:
            if lane.position_type == -1:
                detect_range = lane.startpoint[1]
            if lane.position_type == -2:
                self.near_lane_infor[0] == 1
            elif lane.position_type == 2:
                self.near_lane_infor[1] == 1

        for obstacle in self.obstacles:  # 更新周围汽车信息
            if obstacle.lane == 0 and obstacle.position[1] + obstacle.position[3] >= detect_range:
                if obstacle.speed[1] <= -1:
                    self.speed_infor = 1
                elif obstacle.speed[1] >= -1:
                    self.speed_infor = 0
                if obstacle.position[1] + obstacle.position[3] <= 420:
                    self.ego_lane_infor = 1
                else:
                    self.ego_lane_infor = 2

            elif obstacle.lane == -1 and obstacle.position[1] + obstacle.position[3] >= 420:
                self.near_car_infor[0] = 1

            elif obstacle.lane == 1 and obstacle.position[1] + obstacle.position[3] >= 420:
                self.near_car_infor[1] = 1

    def update_trigger(self, state_now, obstacles, lane):
        self.update_infor(obstacles, lane)

        self.state_now = state_now
        if self.state_now == 'Cruise' and self.ego_lane_infor == 1 and self.speed_infor == 0:  # 发现前车且速度差为负
            self.state_trigger = 'Cruise2Follow'
        elif self.state_now == 'Follow' and (self.ego_lane_infor == 0 or (self.ego_lane_infor == 1 and self.speed_infor == 1)):
            self.state_trigger = 'Follow2Cruise'
        elif self.state_now == 'Cruise' and self.ego_lane_infor == 2:
            self.state_trigger = 'Cruise2ACA'
        elif self.state_now == 'Follow' and self.ego_lane_infor == 2:
            self.state_trigger = 'Follow2ACA'
        elif self.state_now == 'ActiveCollisionAvoidance' and self.ego_lane_infor == 1 and self.speed_infor == 0:
            self.state_trigger = 'ACA2Follow'
        elif self.state_now == 'ActiveCollisionAvoidance' and self.ego_lane_infor == 0:
            self.state_trigger = 'ACA2Cruise'
        else:
            self.state_trigger = None
        '''
        if self.state_now == 'Cruise' and self.ego_lane_infor == 1 and self.speed_infor == 0:  # 发现前车且速度差为负
            self.state_trigger = 'Cruise2Follow'
        elif self.state_now == 'Follow' and (self.ego_lane_infor == 0 or (self.ego_lane_infor == 1 and self.speed_infor == 1)):
            self.state_trigger = 'Follow2Cruise'
        elif self.state_now == 'Follow' and self.ego_lane_infor == 2 and self.near_car_infor == [1, 1]:
            self.state_trigger = 'Follow2ACA'
        elif self.state_now == 'ActiveCollisionAvoidance' and self.ego_lane_infor == 1 and self.speed_infor == 0 and self.near_car_infor != [1, 1]:
            self.state_trigger = 'ACA2Follow'
        elif self.state_now == 'Follow' and self.near_car_infor != [1, 1] and self.t_sheld == 1:
            self.state_trigger = 'Follow2Passing'
        elif self.state_now == 'Passing' and self.ego_lane_infor == 1 and (self.near_lane_infor[0] != 0 or self.near_lane_infor[1] != 0):
            self.state_trigger = 'Passing2Follow'
        elif self.state_now == 'Passing' and self.ego_lane_infor == 0:
            self.state_trigger = 'Passing2Cruise'
        elif self.state_now == 'Passing' and self.ego_lane_infor == 2:
            self.state_trigger = 'Passing2ACA'
        elif self.state_now == 'ActiveCollisionAvoidance' and self.ego_lane_infor == 0:
            self.state_trigger = 'ACA2Cruise'
        else:
            self.state_trigger = None
        '''
        return self.state_trigger
        # fsmplanner = FSMPlanner("autotruck")
