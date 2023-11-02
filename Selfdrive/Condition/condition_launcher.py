import time
import os
import sys
from loguru import logger
from shared_memory_dict import SharedMemoryDict

from truck_condition import Truck
from hardware_condition import Hardware

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.append(project_path)
from Common.log import condition_data_level, condition_info_level, condition_warning_level

logger.add(os.path.join(project_path, "log/run.log"), rotation="100 MB")

condition_dict = SharedMemoryDict(name='condition', size=1024)


class Condition:
    def __init__(self):
        self.power = 'on'
        self.response_time = 0.05
        self.truck = Truck()
        self.hardware = Hardware()
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
            
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
        else:
            logger.log("ConditionWarning", "Option dictionary is broken!")
            
    def run(self):
        t0 = time.time()
        
        self.truck.update()
        self.truck.publish()
        
        self.hardware.update()
        self.hardware.publish()
        
        t1 = time.time()
        loop_time = t1 - t0
        logger.log("ConditionInfo", "Loop time: {}", loop_time)
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)
        
        
def main():
    logger.log("ConditionInfo", "Condition core start.")
    c = Condition()
    while True:
        c.update()
        c.run()
        if c.power == 'off':
            logger.log("ConditionInfo", "Condition core end. If process is not killed successfully, please press Ctrl+C to kill process manually.")
            time.sleep(1)
            break
            
        
if __name__ == "__main__":
    main()
    
