import sys
import os
import cv2
import time
import numpy as np
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Condition.speedorc import SpeedOCR
from Common.iodata import load_pkl

speedocr = SpeedOCR()
im0 = None
bar = None
while True:
    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))
    try:
        im0 = np.load(os.path.join(project_path, "temp/screen.npy"))
    except ValueError:
        print('lost screen data')

    if im0 is not None:
        bar = im0[750:768, 545:595, :]  # 截取信息条[18, 480, 3]
        speedocr.update(bar)
        speedocr.publish()

    time.sleep(0.05)
    if option_dict is not None and len(option_dict):
        if option_dict['power'] == 0:
            break
