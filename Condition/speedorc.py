import os
import sys
import re
import string
import pickle
import numpy as np
from paddleocr import PaddleOCR

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Common.iodata import save_pkl


class SpeedOCR:
    def __init__(self):
        self.speed = 0
        self.over_speed = 0
        self.ocr = PaddleOCR(enable_mkldnn=True, use_tensorrt=False, use_angle_cls=False, lang="en", use_gpu=False,
                             show_log=False)

    def update(self, bar):
        text = self.ocr.ocr(bar, cls=False)
        if len(text) > 0:
            if len(text[0]) > 0:
                speed = text[0][0][1][0]
                speed = re.sub('[a-zA-Z]', '', speed)
                for i in string.punctuation:
                    speed = speed.replace(i, '')
                if ' ' in speed:
                    speed = speed.replace(' ', '')
                if speed == '':
                    speed = 0
                self.speed = int(speed) / 3.6
        if np.average(bar[:, :, 0]) >= 110:  # 超速
            self.over_speed = 1
        else:
            self.over_speed = 0

    def publish(self):
        speed_dict = {'speed': self.speed, 'over_speed': self.over_speed}
        save_pkl(os.path.join(project_path, 'temp/speed.pkl'), speed_dict)
