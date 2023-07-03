from Screen.grab_screen import grab_screen
import cv2
import time
import os

index = 1159
dataset_path = '../datasets/speed_dataset'
time.sleep(10)
start_time = time.time()
while True:
    end_time = time.time()
    if end_time - start_time >= 0.2:
        img = grab_screen()
        bar = cv2.cvtColor(img[750:768, 545:595, :], cv2.COLOR_RGB2BGR)  # 截取信息条[18, 50, 3]
        img_name = '{:0>4d}'.format(int(index))
        image_path = os.path.join(dataset_path, 'images', img_name) + '.jpg'
        print(image_path)
        cv2.imwrite(image_path, bar)
        cv2.imshow('bar', bar)
        index = index + 1
        start_time = time.time()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
