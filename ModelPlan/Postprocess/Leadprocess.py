import numpy as np

LEAD_MHP_SELECTION = 3
LEAD_MHP_N = 2
LEAD_TRAJ_LEN = 6


class Leads:
    def __init__(self, lead_lines):
        self.future_prob = lead_lines[-3:]
        self.leader1 = LeaderPrediction(lead_lines[:51])
        self.leader2 = LeaderPrediction(lead_lines[51:102])
        self.leaders = [self.leader1, self.leader2]
        self.conf = 0.5

    def select(self):
        leaders = []
        for leader in self.leaders:
            if leader.prob[0] >= self.conf:
                leaders.append(leader.mean)
        leaders = np.array(leaders)
        return leaders


class LeaderPrediction:
    def __init__(self, line):
        self.mean = np.concatenate((line[0:24:4].reshape((1, -1)),  # x
                                    line[1:24:4].reshape((1, -1)),  # y
                                    np.zeros((1, 6)),  # z
                                    line[2:24:4].reshape((1, -1)),  # 速度
                                    line[3:24:4].reshape((1, -1))), axis=0).T  # 加速度
        self.std = np.concatenate((line[24:48:4].reshape((1, -1)),
                                   line[25:48:4].reshape((1, -1)),
                                   np.zeros((1, 6)),
                                   line[26:48:4].reshape((1, -1)),
                                   line[27:48:4].reshape((1, -1))), axis=0).T
        self.prob = line[-3:]
