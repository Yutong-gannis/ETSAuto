import numpy as np

TRAJECTORY_SIZE = 33
X_IDXS = np.array([[0., 0.1875, 0.75, 1.6875, 3., 4.6875,
                    6.75, 9.1875, 12., 15.1875, 18.75, 22.6875,
                    27., 31.6875, 36.75, 42.1875, 48., 54.1875,
                    60.75, 67.6875, 75., 82.6875, 90.75, 99.1875,
                    108., 117.1875, 126.75, 136.6875, 147., 157.6875,
                    168.75, 180.1875, 192.]])


class Edges:
    def __init__(self, edge_lines):
        mean_line = edge_lines[:132]
        std_line = edge_lines[132:]
        self.mean = EdgePrediction(mean_line)
        self.std = EdgePrediction(std_line)

    def extract(self):
        return self.mean.edges


class EdgePrediction:
    def __init__(self, line):
        self.line_length = 66
        edge1 = np.concatenate((X_IDXS, line[0:self.line_length:2].reshape((1, -1)), line[1:self.line_length:2].reshape((1, -1))), axis=0).T
        edge2 = np.concatenate((X_IDXS, line[self.line_length:self.line_length * 2:2].reshape((1, -1)), line[self.line_length + 1:self.line_length * 2:2].reshape((1, -1))), axis=0).T
        self.edges = [edge1, edge2]
