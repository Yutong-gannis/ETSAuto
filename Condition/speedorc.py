import os
import re
import string
import numpy as np
from paddleocr import PaddleOCR


class SpeedOCR:
    def __init__(self):
        self.speed = 0
        self.over_speed = 0
        self.ocr = PaddleOCR(enable_mkldnn=True, use_tensorrt=True, use_angle_cls=False, lang="en", use_gpu=False,
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

    def publish(self, project_path):
        publish_data = [self.speed, self.over_speed]
        print(publish_data)
        f = open(os.path.join(project_path, "temp/speed.txt"), "w")
        for element in publish_data:
            f.write(str(element) + ' ')
        f.close()
