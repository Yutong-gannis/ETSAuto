class Pose:
    def __init__(self, line):
        self.velocity_mean = line[:3]
        self.rotation_mean = line[3:6]
        self.velocity_std = line[6:9]
        self.rotation_std = line[9:12]
