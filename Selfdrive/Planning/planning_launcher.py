import time
import os
import sys
import numpy as np
from shared_memory_dict import SharedMemoryDict
from loguru import logger

from planner import Planner

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.append(project_path)
from Common.log import planning_data_level, planning_info_level, planning_warning_level


logger.add(os.path.join(project_path, "log/run.log"), rotation="100 MB")

plan_dict = SharedMemoryDict(name='plan', size=1024)
states_dict = SharedMemoryDict(name='states', size=1024)
fcw_dict = SharedMemoryDict(name='fcw', size=1024)


class Planning:
    def __init__(self):
        self.power = 'on'
        self.planner = Planner()
        self.fps = 0.02
        self.response_time = 0.05
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
        else:
            logger.log("PlanningWarning", "Option dictionary is broken!")
        
    def run(self):
        t0 = time.time()
        
        self.planner.update()
        self.planner.run()
        t1 = time.time()
        
        loop_time = t1 - t0
        logger.log("PlanningInfo", "Loop time: {}", loop_time)
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)

def main():
    logger.log("PlanningInfo", "Planning core start.")
    p = Planning()
    while True:
        p.update()
        p.run()
        if p.power == 'off':
            time.sleep(1)
            break
        
if __name__ == "__main__":
    main()
