import numpy as np
import math
import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
sys.path.insert(0, os.path.join(current_path, '../../Camera'))
import orientation

FULL_FRAME_SIZE = [1360, 768]
W, H = FULL_FRAME_SIZE[0], FULL_FRAME_SIZE[1]
eon_focal_length = FOCAL = 910.0

# aka 'K' aka camera_frame_from_view_frame
eon_intrinsics = np.array([
    [1200, 0., W / 2.],
    [0., 300, H / 2.],
    [0., 0., 1.]])


def transform_img(base_img,
                  augment_trans=np.array([0, 0, 0]),
                  augment_eulers=np.array([0, 0, 0]),
                  from_intr=eon_intrinsics,
                  to_intr=eon_intrinsics,
                  output_size=None,
                  pretransform=None,
                  top_hacks=False,
                  yuv=False,
                  alpha=1.0,
                  beta=0,
                  blur=0):
    import cv2  # pylint: disable=import-error
    cv2.setNumThreads(1)

    if yuv:
        base_img = cv2.cvtColor(base_img, cv2.COLOR_YUV2RGB_I420)

    size = base_img.shape[:2]
    if not output_size:
        output_size = size[::-1]

    cy = from_intr[1, 2]

    def get_M(h=1.22):
        quadrangle = np.array([[0, cy + 20],
                               [size[1] - 1, cy + 20],
                               [0, size[0] - 1],
                               [size[1] - 1, size[0] - 1]], dtype=np.float32)
        quadrangle_norm = np.hstack((normalize(quadrangle, intrinsics=from_intr), np.ones((4, 1))))
        quadrangle_world = np.column_stack((h * quadrangle_norm[:, 0] / quadrangle_norm[:, 1],
                                            h * np.ones(4),
                                            h / quadrangle_norm[:, 1]))
        rot = orientation.rot_from_euler(augment_eulers)
        to_extrinsics = np.hstack((rot.T, -augment_trans[:, None]))
        to_KE = to_intr.dot(to_extrinsics)
        warped_quadrangle_full = np.einsum('jk,ik->ij', to_KE, np.hstack((quadrangle_world, np.ones((4, 1)))))
        warped_quadrangle = np.column_stack((warped_quadrangle_full[:, 0] / warped_quadrangle_full[:, 2],
                                             warped_quadrangle_full[:, 1] / warped_quadrangle_full[:, 2])).astype(
            np.float32)
        M = cv2.getPerspectiveTransform(quadrangle, warped_quadrangle.astype(np.float32))
        return M

    M = get_M()
    if pretransform is not None:
        M = M.dot(pretransform)
    augmented_rgb = cv2.warpPerspective(base_img, M, output_size, borderMode=cv2.BORDER_REPLICATE)

    if top_hacks:
        cyy = int(math.ceil(to_intr[1, 2]))
        M = get_M(1000)
        if pretransform is not None:
            M = M.dot(pretransform)
        augmented_rgb[:cyy] = cv2.warpPerspective(base_img, M, (output_size[0], cyy), borderMode=cv2.BORDER_REPLICATE)

    # brightness and contrast augment
    augmented_rgb = np.clip((float(alpha) * augmented_rgb + beta), 0, 255).astype(np.uint8)

    # gaussian blur
    if blur > 0:
        augmented_rgb = cv2.GaussianBlur(augmented_rgb, (blur * 2 + 1, blur * 2 + 1), cv2.BORDER_DEFAULT)

    if yuv:
        augmented_img = cv2.cvtColor(augmented_rgb, cv2.COLOR_RGB2YUV_I420)
    else:
        augmented_img = augmented_rgb
    return augmented_img


def normalize(img_pts, intrinsics=eon_intrinsics):
    # normalizes image coordinates
    # accepts single pt or array of pts
    intrinsics_inv = np.linalg.inv(intrinsics)
    img_pts = np.array(img_pts)
    input_shape = img_pts.shape
    img_pts = np.atleast_2d(img_pts)
    img_pts = np.hstack((img_pts, np.ones((img_pts.shape[0], 1))))
    img_pts_normalized = img_pts.dot(intrinsics_inv.T)
    img_pts_normalized[(img_pts < 0).any(axis=1)] = np.nan
    return img_pts_normalized[:, :2].reshape(input_shape)