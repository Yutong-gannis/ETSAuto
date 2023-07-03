import numpy as np

FULL_FRAME_SIZE = [1360, 768]
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

eon_intrinsics = np.array([
    [1000, 0., W / 2.],
    [0., 350, H / 2. + 5],
    [0., 0., 1.]])
