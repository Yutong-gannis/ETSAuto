import numpy as np
import scipy

NUM_PLAN = 5
PLAN_LENGTH = 991
TRAJECTORT_SIZE = 33


class Plans:
    def __init__(self, plan_line):
        self.num_plan = NUM_PLAN
        self.plan_length = PLAN_LENGTH
        self.plan_0 = PlanPrediction(plan_line[0: self.plan_length])
        self.plan_1 = PlanPrediction(plan_line[self.plan_length: self.plan_length * 2])
        self.plan_2 = PlanPrediction(plan_line[self.plan_length * 2: self.plan_length * 3])
        self.plan_3 = PlanPrediction(plan_line[self.plan_length * 3: self.plan_length * 4])
        self.plan_4 = PlanPrediction(plan_line[self.plan_length * 4: self.plan_length * 5])
        self.plans = [self.plan_0, self.plan_1, self.plan_2, self.plan_3, self.plan_4]

    def select(self):
        plan, plan_position, plan_velocity, plan_acc, plan_yaw, plan_yaw_rate, plan_angle = None, None, None, None, None, None, None
        for i in range(self.num_plan):
            if plan is None:
                plan = self.plans[i]
            else:
                if self.plans[i].prob > plan.prob:
                    plan = self.plans[i]

        if plan is not None:
            plan_position = []
            plan_velocity = []
            plan_acc = []
            plan_yaw = []
            plan_yaw_rate = []
            plan_angle = []
            for i in range(TRAJECTORT_SIZE):
                position = plan.mean[i].position[0:3]
                velocity = plan.mean[i].velocity[0:3]  # 米每秒
                acc = plan.mean[i].acceleration[0:2]
                yaw = plan.mean[i].rotation[0:3]
                yaw_rate = plan.mean[i].rotation_rate[0:3]
                acc_total = (acc[0]**2 + acc[1]**2)**0.5
                angle = np.arctan(acc[1]/acc[0])

                plan_position.append(position)
                plan_velocity.append(velocity)
                plan_acc.append(acc_total)
                plan_yaw.append(yaw)
                plan_yaw_rate.append(yaw_rate)
                plan_angle.append(angle)
            plan_position = np.array(plan_position)
            plan_velocity = np.array(plan_velocity)
            plan_acc = np.array(plan_acc)
            plan_yaw = np.array(plan_yaw)
            plan_yaw_rate = np.array(plan_yaw_rate)
            plan_angle = np.array(plan_angle)

        return plan_position, plan_velocity, plan_acc, plan_yaw, plan_yaw_rate, plan_angle


class PlanPrediction:
    def __init__(self, plan_i):
        self.trajectory_size = TRAJECTORT_SIZE
        self.mean = []
        self.std = []
        for i in range(self.trajectory_size):
            self.mean.append(PlanElement(plan_i[i*15: (i + 1)*15]))
            self.std.append(PlanElement(plan_i[i*15 + self.trajectory_size*15: (i+1)*15 + self.trajectory_size*15]))
        self.prob = plan_i[-1]


class PlanElement:
    def __init__(self, path_i):
        self.position = path_i[0:3]
        self.velocity = path_i[3:6]
        self.acceleration = path_i[6:9]
        self.rotation = path_i[9:12]
        self.rotation_rate = path_i[12:15]


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x
