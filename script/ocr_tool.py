from paddleocr import PaddleOCR
import re
import string

def speed_detect(ocr, bar, speed_last):
    text = ocr.ocr(bar, cls=False)
    if len(text) > 0:
        if len(text[0]) > 0:
            speed = text[0][0][1][0]
            speed = re.sub('[a-zA-Z]', '', speed)
            for i in string.punctuation:
                speed = speed.replace(i, '')
            if ' ' in speed: speed = speed.replace(' ', '')
            if speed == '': speed = 0
            speed = int(speed)
            return speed
        return speed_last
    return speed_last