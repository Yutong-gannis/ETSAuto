from Screen.grab_screen import grab_screen
import cv2
import time
import os
import win32api

index = 158
dataset_path = '../datasets/nav_dataset'
flag = 0
start_time = time.time()
while True:
    if win32api.GetAsyncKeyState(0x61):
        flag = 1
    elif win32api.GetAsyncKeyState(0x62):
        flag = 0
    end_time = time.time()
    if end_time - start_time >= 1:
        img = grab_screen()
        nav = cv2.cvtColor(img[610:740, 580:780, :], cv2.COLOR_RGB2BGR)
        cv2.imshow('nav', nav)
        img_name = '{:0>4d}'.format(int(index))
        image_path = os.path.join(dataset_path, 'images', img_name) + '.jpg'
        print(flag)
        if flag == 1:
            cv2.imwrite(image_path, nav)
            print(image_path)
            index = index + 1
        start_time = time.time()
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
