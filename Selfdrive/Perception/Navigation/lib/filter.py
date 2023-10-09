import cv2


def filter_out_red(img):
    img_blue = img[:, :, 0]
    _, mask_blue = cv2.threshold(img_blue, 180, 125, cv2.THRESH_BINARY)
    img_red = img[:, :, 2]
    _, mask_red = cv2.threshold(img_red, 100, 125, cv2.THRESH_BINARY_INV)
    mask = mask_blue + mask_red
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    mask = cv2.dilate(mask, kernel, iterations=6)
    mask = cv2.Canny(mask, 50, 150)
    _, mask = cv2.threshold(mask, 180, 180, cv2.THRESH_BINARY)
    mask = mask+75
    return mask