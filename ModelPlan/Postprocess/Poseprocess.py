import numpy as np


class Pose:
    def __init__(self, line):
        self.velocity_mean = line[:3]
        self.rotation_mean = line[3:6]
        self.velocity_std = line[6:9]
        self.rotation_std = line[9:12]

    def extract(self):
        pose = np.concatenate((self.velocity_mean.reshape(1, -1), self.rotation_mean.reshape(1, -1)), axis=0)
        print(pose)
        return pose
