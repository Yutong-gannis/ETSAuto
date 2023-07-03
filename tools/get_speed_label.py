import warnings
import os
import sys
import cv2
from paddleocr import PaddleOCR

warnings.filterwarnings('ignore')

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

condition_path = os.path.abspath(os.path.join(project_path, 'Condition'))
sys.path.insert(0, condition_path)
from speedorc import speed_detect


def get_imlist(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]


ocr = PaddleOCR(enable_mkldnn=True, use_tensorrt=True, use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
dataset_path = '../datasets/speed_dataset'
index = 0
img_paths = get_imlist(os.path.join(dataset_path, 'images'))
for img_path in img_paths:
    index = img_path[-8:-4]
    print(index)
    bar = cv2.imread(img_path)
    speed, over_speed = speed_detect(ocr, bar)
    cv2.imshow('bar', bar)
    label_path = os.path.join(dataset_path, 'labels', index) + '.txt'
    print(speed)
    with open(label_path, "w") as f:
        f.write(str(speed))
    # ctrl+Q退出
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
