import numpy as np

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
        self.conf = -20

    def select(self):
        plan, plan_position, plan_velocity, plan_acc, plan_angle = None, None, None, None, None
        for i in range(self.num_plan):
            if self.plans[i].prob > self.conf:
                if plan is None:
                    plan = self.plans[i]
                else:
                    if self.plans[i].prob > plan.prob:
                        plan = self.plans[i]

        plan_position = []
        plan_velocity = []
        plan_acc = []
        plan_angle = []
        if plan is not None:
            for i in range(TRAJECTORT_SIZE):
                plan_position.append(plan.mean[i].position[0:2])
                velocity = plan.mean[i].velocity[0:2] * 3.6  # 千米每秒
                acc = plan.mean[i].acceleration[0:2]
                velocity_total = (velocity[0]**2 + velocity[1]**2)**0.5
                acc_total = (acc[0]**2 + acc[1]**2)**0.5
                angle = np.arctan(velocity[1]/velocity[0])
                plan_velocity.append(velocity_total)
                plan_acc.append(acc_total)
                plan_angle.append(angle)
        plan_position = np.array(plan_position)
        plan_velocity = np.array(plan_velocity)
        plan_acc = np.array(plan_acc)
        plan_angle = np.array(plan_angle)
        return plan_position, plan_velocity, plan_acc, plan_angle


class PlanPrediction:
    def __init__(self, plan_i):
        self.tranjectory_size = TRAJECTORT_SIZE
        self.mean = []
        self.std = []
        for i in range(self.tranjectory_size):
            self.mean.append(PlanElement(plan_i[i*15: (i + 1)*15]))
            self.std.append(PlanElement(plan_i[i*15 + self.tranjectory_size*15: (i+1)*15 + self.tranjectory_size*15]))
        self.prob = plan_i[-1]


class PlanElement:
    def __init__(self, path_i):
        self.position = path_i[0:3]
        self.velocity = path_i[3:6]
        self.acceleration = path_i[6:9]
        self.rotation = path_i[9:12]
        self.rotation_rate = path_i[12:15]
