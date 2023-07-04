import numpy as np

DISENGAGE_LEN = 5
BLINKER_LEN = 6
DESIRE_PRED_LEN = 4
DESIRE_LEN = 8
FCW_THRESHOLD_5MS2_HIGH = 0.15
FCW_THRESHOLD_5MS2_LOW = 0.05
FCW_THRESHOLD_3MS2 = 0.7


class ModelOutputMeta:
    def __init__(self, line):
        self.prev_brake_5ms2_probs = [0, 0, 0, 0, 0]
        self.prev_brake_3ms2_probs = [0, 0, 0]
        self.fcw_threshold = True
        self.lat_long_t = [2, 4, 6, 8, 10]

        self.desire_state_prob = softmax(line[:8])
        self.engaged_prob = line[8]
        self.disengage_prob = sigmoid(line[9:44].reshape((7, 5)))  # 脱离操作：油门、刹车、方向盘、3m/s2的刹车、4m/s2的刹车、5m/s2的刹车，踩油门
        self.blinker_prob = line[44:56]
        self.desire_pred_prob = line[56:88].reshape((4, 8))

        for i in range(DESIRE_PRED_LEN):
            self.desire_pred_prob[i, :] = softmax(self.desire_pred_prob[i, :])

        self.prev_brake_5ms2_probs.pop(0)
        self.prev_brake_3ms2_probs.pop(0)

        self.prev_brake_5ms2_probs.append(self.disengage_prob[5, 0])
        self.prev_brake_3ms2_probs.append(self.disengage_prob[3, 0])

        for i in range(len(self.prev_brake_5ms2_probs)):
            threshold = FCW_THRESHOLD_5MS2_LOW if i < 2 else FCW_THRESHOLD_5MS2_HIGH
            self.fcw_threshold = self.fcw_threshold and self.prev_brake_5ms2_probs[i] > threshold

        for i in range(len(self.prev_brake_3ms2_probs)):
            self.fcw_threshold = self.fcw_threshold and self.prev_brake_3ms2_probs[i] > FCW_THRESHOLD_3MS2


def sigmoid(x):
    x = 1 / (1 + np.exp(-x))
    return x


def softmax(x):
    x -= np.max(x)
    return np.exp(x) / np.sum(np.exp(x))
