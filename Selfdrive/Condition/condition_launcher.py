import os
import sys
import time
import cv2

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Message.iodata import load_pkl
from lib.ets2sdktelemetry import Ets2SdkTelemetry
from lib.sharedmemory import SharedMemory
from truck_condition import Truck

class Condition:
    def __init__(self):
        self.power = 'on'
        self.response_time = 0.05
        self.truck = Truck()
        
    def update(self):
        option_dict = load_pkl(os.path.join(project_path, 'Message/temp/option.pkl'))
        if option_dict is not None:
            self.power = option_dict['power']
            
    def run(self):
        self.truck.update()
        self.truck.publish()
        time.sleep(self.response_time)
        
        
def main():
    time.sleep(10)
    c = Condition()
    while True:
        c.update()
        c.run()
        if c.power == 'off':
            time.sleep(1)
            break
        
if __name__ == "__main__":
    main()