import numpy as np


def parse_image(frame):
    H = (frame.shape[0] * 2) // 3
    W = frame.shape[1]
    parsed = np.zeros((6, H // 2, W // 2), dtype=np.uint8)

    parsed[0] = frame[0:H:2, 0::2]
    parsed[1] = frame[1:H:2, 0::2]
    parsed[2] = frame[0:H:2, 1::2]
    parsed[3] = frame[1:H:2, 1::2]
    parsed[4] = frame[H:H + H // 4].reshape((-1, H // 2, W // 2))
    parsed[5] = frame[H + H // 4:H + H // 2].reshape((-1, H // 2, W // 2))

    return parsed
