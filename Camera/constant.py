import numpy as np
import Camera.orientation as orientation

device_frame_from_view_frame = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.]
])
view_frame_from_device_frame = device_frame_from_view_frame.T


def get_view_frame_from_calib_frame(roll, pitch, yaw, height):
    device_from_calib = orientation.rot_from_euler([roll, pitch, yaw])
    view_from_calib = view_frame_from_device_frame.dot(device_from_calib)
    return np.hstack((view_from_calib, [[0], [height], [0]]))


def get_view_frame_from_road_frame(roll, pitch, yaw, height):
    device_from_road = orientation.rot_from_euler([roll, pitch, yaw]).dot(np.diag([1, -1, -1]))
    view_from_road = view_frame_from_device_frame.dot(device_from_road)
    return np.hstack((view_from_road, [[0], [height], [0]]))


plot_img_width = 1360
plot_img_height = 768

FULL_FRAME_SIZE = (1360, 768)
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 900

zoom = FULL_FRAME_SIZE[0] / plot_img_width
CALIB_BB_TO_FULL = np.asarray([
    [zoom, 0., 0.],
    [0., zoom, 0.],
    [0., 0., 1.]])

# MED model
MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_YUV_SIZE = (MEDMODEL_INPUT_SIZE[0], MEDMODEL_INPUT_SIZE[1] * 3 // 2)
MEDMODEL_CY = 47.6

medmodel_fl = 910.0
medmodel_intrinsics = np.array([
    [medmodel_fl, 0.0, 0.5 * MEDMODEL_INPUT_SIZE[0]],
    [0.0, medmodel_fl, MEDMODEL_CY],
    [0.0, 0.0, 1.0]])

# BIG model
BIGMODEL_INPUT_SIZE = (1024, 512)
BIGMODEL_YUV_SIZE = (BIGMODEL_INPUT_SIZE[0], BIGMODEL_INPUT_SIZE[1] * 3 // 2)

bigmodel_fl = 910.0
bigmodel_intrinsics = np.array([
    [bigmodel_fl, 0.0, 0.5 * BIGMODEL_INPUT_SIZE[0]],
    [0.0, bigmodel_fl, 256 + MEDMODEL_CY],
    [0.0, 0.0, 1.0]])

# SBIG model (big model with the size of small model)
SBIGMODEL_INPUT_SIZE = (512, 256)
SBIGMODEL_YUV_SIZE = (SBIGMODEL_INPUT_SIZE[0], SBIGMODEL_INPUT_SIZE[1] * 3 // 2)

sbigmodel_fl = 455.0
sbigmodel_intrinsics = np.array([
    [sbigmodel_fl, 0.0, 0.5 * SBIGMODEL_INPUT_SIZE[0]],
    [0.0, sbigmodel_fl, 0.5 * (256 + MEDMODEL_CY)],
    [0.0, 0.0, 1.0]])

bigmodel_frame_from_calib_frame = np.dot(bigmodel_intrinsics,
                                         get_view_frame_from_calib_frame(0, 0, 0, 0))

sbigmodel_frame_from_calib_frame = np.dot(sbigmodel_intrinsics,
                                          get_view_frame_from_calib_frame(0, 0, 0, 0))

medmodel_frame_from_calib_frame = np.dot(medmodel_intrinsics,
                                         get_view_frame_from_calib_frame(0, 0, 0, 0))

medmodel_frame_from_bigmodel_frame = np.dot(medmodel_intrinsics, np.linalg.inv(bigmodel_intrinsics))

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
    [1000, 0., W / 2.],
    [0., 350, H / 2. + 20],
    [0., 0., 1.]])
