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
        best_plan = None
        for i in range(self.num_plan):
            if best_plan is None:
                best_plan = self.plans[i]
            else:
                if self.plans[i].prob > best_plan.prob:
                    best_plan = self.plans[i]
        return best_plan.mean


class PlanPrediction:
    def __init__(self, plan_i):
        self.trajectory_size = TRAJECTORT_SIZE
        self.mean = plan_i[0:495].reshape((33, 5, 3))  # 位置，速度，加速度，旋转角，旋转角速度
        self.std = plan_i[495:990].reshape((33, 5, 3))
        self.prob = plan_i[-1]
