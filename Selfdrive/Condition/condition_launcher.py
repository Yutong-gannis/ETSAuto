import os
import sys
import time
import cv2
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from lib.ets2sdktelemetry import Ets2SdkTelemetry
from lib.sharedmemory import SharedMemory
from truck_condition import Truck


condition_dict = SharedMemoryDict(name='condition', size=1024)


class Condition:
    def __init__(self):
        self.power = 'on'
        self.response_time = 0.05
        self.truck = Truck()
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
            
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
            
    def run(self):
        t0 = time.time()
        
        self.truck.update()
        self.truck.publish()
        
        t1 = time.time()
        loop_time = t1 - t0
        print(loop_time)
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)
        
        
def main():
    c = Condition()
    while True:
        c.update()
        c.run()
        if c.power == 'off':
            time.sleep(1)
            break
        
if __name__ == "__main__":
    main()