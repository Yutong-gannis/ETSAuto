import numpy as np

LEAD_MHP_SELECTION = 3
LEAD_MHP_N = 2
LEAD_TRAJ_LEN = 6


class Leads:
    def __init__(self, lead_lines):
        self.leader1 = LeaderPrediction(lead_lines[:48])
        self.leader2 = LeaderPrediction(lead_lines[51:99])
        self.leaders = [self.leader1, self.leader2]
        self.prob_1 = sigmoid(lead_lines[48:51])
        self.prob_2 = sigmoid(lead_lines[99:102])
        self.future_prob = sigmoid(lead_lines[-3:])
        self.conf = 0.5

    def select(self):
        leader = None
        if self.future_prob[0] >= self.conf:
            if self.prob_1[0] >= self.prob_2[0]:
                leader = self.leader1.mean
            else:
                leader = self.leader2.mean
        return leader


class LeaderPrediction:
    def __init__(self, line):
        self.mean = np.concatenate((line[0:24:4].reshape((1, -1)),  # x
                                    line[1:24:4].reshape((1, -1)),  # y
                                    np.ones((1, 6))*(-1),  # z
                                    line[2:24:4].reshape((1, -1)),  # 速度
                                    line[3:24:4].reshape((1, -1))), axis=0).T  # 加速度
        self.std = np.concatenate((line[24:48:4].reshape((1, -1)),
                                   line[25:48:4].reshape((1, -1)),
                                   np.ones((1, 6))*(-1),
                                   line[26:48:4].reshape((1, -1)),
                                   line[27:48:4].reshape((1, -1))), axis=0).T


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x
