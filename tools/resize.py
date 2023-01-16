import cv2

img = cv2.imread(r"D:/autodrive/script/test_Moment.jpg")
img = cv2.resize(img, (1280, 720))
cv2.imwrite(r"D:/autodrive/script/test_Moment_small.jpg", img)