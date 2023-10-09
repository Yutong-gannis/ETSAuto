import warnings
import os
import sys
import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from shared_memory_dict import SharedMemoryDict

warnings.filterwarnings('ignore')
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Perception.LaneDetection.lanedetector import Bev_Lanedet
from Perception.ObjectDetection.objectdetector import YOLOv8
from Perception.Screen.grab_screen import ScreenGraber
from Perception.Navigation.Navigation_Process import NavProcess


lane_dict = SharedMemoryDict(name='lane', size=12000)
dets_dict = SharedMemoryDict(name='dets', size=1024)
nav_dict = SharedMemoryDict(name='nav', size=1024)


class Perception:
    """The main class of perception
    """
    def __init__(self):
        self.power = 'on'
        self.response_time = 0.05
        lane_path = os.path.abspath(os.path.join(project_path, '../weights/bevlanedet/resnet18_0.5_v4/ep002.onnx'))
        yolo_path = os.path.abspath(os.path.join(project_path, '../weights/yolov8/best.onnx'))
        self.objectdetector = Bev_Lanedet(lane_path)
        self.lanedetector = YOLOv8(yolo_path)
        self.screengraber = ScreenGraber()
        self.navprocess = NavProcess()
        self.frame = self.screengraber.update()
        self.fps = 0
        self.frequency = 0
    
    def update(self):
        """Update the state of perception
        """
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
            
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
        self.count()
        
    def count(self):
        self.frequency = self.frequency + 1
        
    def run(self):
        """Run the perceptions
        """
        start_time = time.time()
        #pool = ThreadPoolExecutor(max_workers=3)
        #thread1 = pool.submit(self.lanedetector.infer, self.frame)
        #thread2 = pool.submit(self.screengraber.update)
        #thread3 = pool.submit(self.objectdetector.infer, self.frame)
        self.lanedetector.infer(self.frame)
        self.frame = self.screengraber.update()
        self.objectdetector.infer(self.frame)
        self.navprocess.run(self.frame)
        #self.frame = thread2.result()
        #pool.shutdown()
        end_time = time.time()
        self.fps = end_time - start_time
        

def main():
    p = Perception()
    while True:
        p.update()
        p.run()
        
        if p.power == 'off':
            time.sleep(1)
            break
        
if __name__ == "__main__":
    main()
    