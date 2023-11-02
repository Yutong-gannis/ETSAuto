import os
import sys
import time
import cv2
from loguru import logger
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from shared_memory_dict import SharedMemoryDict

from LaneDetection.lanedetector import Bev_Lanedet
from ObjectDetection.objectdetector import YOLOv8
from Screen.grab_screen import ScreenGraber
from Navigation.Navigation_Process import NavProcess

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.append(project_path)
from Common.log import perception_data_level, perception_error_level, perception_info_level, perception_warning_level


logger.add(os.path.join(project_path, "log/run.log"), rotation="100 MB")

lane_dict = SharedMemoryDict(name='lane', size=12000)
dets_dict = SharedMemoryDict(name='dets', size=1024)
nav_dict = SharedMemoryDict(name='nav', size=1024)


class Perception:
    """The main class of perception
    """
    def __init__(self):
        self.power = 'on'
        self.response_time = 0.05
        lane_path = os.path.abspath(os.path.join(project_path, 'weights/bevlanedet.onnx'))
        yolo_path = os.path.abspath(os.path.join(project_path, 'weights/yolov8n.onnx'))
        self.objectdetector = Bev_Lanedet(lane_path)
        self.lanedetector = YOLOv8(yolo_path)
        self.screengraber = ScreenGraber()
        self.navprocess = NavProcess()
        self.frame = self.screengraber.update()
        self.response_time = 0.05
    
    def update(self):
        """Update the state of perception
        """
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
            
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
        else:
            logger.log("PerceptionWarning", "Option dictionary is broken!")
        
    def run(self):
        """Run the perceptions
        """
        t0 = time.time()
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
        t1 = time.time()
        loop_time = t1 - t0
        logger.log("PerceptionInfo", "Loop time: {}", loop_time)
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)
        

def main():
    logger.log("PerceptionInfo", "Perception core start.")
    p = Perception()
    while True:
        p.update()
        p.run()
        
        if p.power == 'off':
            logger.log("PerceptionInfo", "Perception core end.")
            time.sleep(1)
            break
        
if __name__ == "__main__":
    main()
    
