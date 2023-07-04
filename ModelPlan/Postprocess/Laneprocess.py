import numpy as np

TRAJECTORY_SIZE = 33
X_IDXS = np.array([[0., 0.1875, 0.75, 1.6875, 3., 4.6875,
                    6.75, 9.1875, 12., 15.1875, 18.75, 22.6875,
                    27., 31.6875, 36.75, 42.1875, 48., 54.1875,
                    60.75, 67.6875, 75., 82.6875, 90.75, 99.1875,
                    108., 117.1875, 126.75, 136.6875, 147., 157.6875,
                    168.75, 180.1875, 192.]])


class Lanes:
    def __init__(self, lanes_line):
        self.mean = LinePrediction(lanes_line[:264])
        self.std = LinePrediction(lanes_line[264:])
        prob = lanes_line[-8:].reshape((4, 2)).T
        self.prob = sigmoid(prob)
        self.conf = 0.3

    def select(self):
        lanes = []
        for i in range(4):
            if self.prob[0, i] >= self.conf:
                lanes.append(self.mean.lines[i])
        return lanes


class LinePrediction:
    def __init__(self, line):
        self.line_length = 66
        line1 = np.concatenate((X_IDXS, line[0:self.line_length].reshape((-1, 2)).T), axis=0).T
        line2 = np.concatenate((X_IDXS, line[self.line_length:self.line_length * 2].reshape((-1, 2)).T), axis=0).T
        line3 = np.concatenate((X_IDXS, line[self.line_length * 2:self.line_length * 3].reshape((-1, 2)).T), axis=0).T
        line4 = np.concatenate((X_IDXS, line[self.line_length * 3:self.line_length * 4].reshape((-1, 2)).T), axis=0).T
        self.lines = [line1, line2, line3, line4]


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x
