import numpy as np
import re
import string

def speed_detect(ocr, bar, truck):
    if np.average(bar[:, :, 2]) >= 120:  # 超速
        truck.over_speed = 1
    else:
        truck.over_speed = 0
    text = ocr.ocr(bar, cls=False)
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
            speed = int(speed)
            truck.dv = speed - truck.speed
            truck.speed = speed
    return truck