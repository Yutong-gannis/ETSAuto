import os
import numpy as np
import time
from loguru import logger
from shared_memory_dict import SharedMemoryDict

from controller import Controller

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
from Common.log import control_data_level, control_info_level, control_warning_level

logger.add(os.path.join(project_path, "log/run.log"), rotation="100 MB")


class Control:
    def __init__(self):
        self.power = 'on'
        self.controller = Controller()
        self.response_time = 0.05
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
            
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
        else:
            logger.log("ControlWarning", "Option dictionary is broken!")
            
    def run(self):
        t0 = time.time()
        self.controller.update()
        self.controller.run()
        t1 = time.time()
        loop_time = t1 - t0
        logger.log("ControlInfo", "Loop time: {}", loop_time)
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)
        
    def exit(self):
        self.controller.exit()


def main():
    logger.log("ControlInfo", "Control core start.")
    time.sleep(10)
    c = Control()
    while True:
        c.update()
        c.run()
        if c.power == 'off':
            logger.log("ControlInfo", "Control core end.")
            c.exit()
            time.sleep(1)
            break

if __name__ == "__main__":
    main()