import warnings
import os
import sys
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings('ignore')
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Perception.LaneDetection.lanedetector import Bev_Lanedet
from Perception.ObjectDetection.objectdetector import YOLOv8
from Perception.Screen.grab_screen import ScreenGraber
from Perception.Navigation.Navigation_Process import nav_process
from Message.iodata import load_pkl, save_pkl


class Perception:
    """The main class of perception
    """
    def __init__(self):
        self.power = 'on'
        self.response_time = 0.05
        lane_path = os.path.abspath(os.path.join(project_path, '../weights/bevlanedet/resnet18_0.5_v2/ep049.onnx'))
        yolo_path = os.path.abspath(os.path.join(project_path, '../weights/yolov8/best.onnx'))
        self.objectdetector = Bev_Lanedet(lane_path)
        self.lanedetector = YOLOv8(yolo_path)
        self.screengraber = ScreenGraber()
        self.frame = self.screengraber.update()
        self.fps = 0
        self.frequency = 0
    
    def update(self):
        """Update the state of perception
        """
        option_dict = load_pkl(os.path.join(project_path, 'Message/temp/option.pkl'))
        if option_dict is not None:
            self.power = option_dict['power']
        self.count()
        
    def count(self):
        self.frequency = self.frequency + 1
        
    def run(self):
        """Run the perceptions
        """
        start_time = time.time()
        pool = ThreadPoolExecutor(max_workers=3)
        thread1 = pool.submit(self.lanedetector.infer, self.frame)
        thread2 = pool.submit(self.screengraber.update)
        if self.frequency % 4 == 0:
            thread3 = pool.submit(self.objectdetector.infer, self.frame)
        self.frame = thread2.result()
        pool.shutdown()
        end_time = time.time()
        self.fps = end_time - start_time
        

def main():
    time.sleep(5)
    p = Perception()
    while True:
        p.update()
        p.run()
    
        if p.power == 'off':
            time.sleep(1)
            break
        
if __name__ == "__main__":
    main()
    
