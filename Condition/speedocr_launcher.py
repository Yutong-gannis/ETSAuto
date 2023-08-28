import sys
import os
import cv2
import time
import numpy as np
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, current_path)
from speedorc import SpeedOCR

speedocr = SpeedOCR()
im0 = None
bar = None
option_list = None
while True:
    try:
        im0 = np.load(os.path.join(project_path, "temp/screen.npy"))
        print(im0.shape)
    except ValueError:
        print('lost screen data')

    if im0 is not None:
        bar = im0[750:768, 545:595, :]  # 截取信息条[18, 480, 3]
        speedocr.update(bar)
        speedocr.publish(project_path)

    time.sleep(0.05)
    try:
        option_list = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
    except ValueError:
        print('lost option data')
    if option_list is not None and len(option_list):
        if option_list[2] == 0:
            break
