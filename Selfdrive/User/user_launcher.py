import sys
import os
import cv2
import numpy as np
import time
from loguru import logger
from shared_memory_dict import SharedMemoryDict

from lib.option import UserOption
from lib.interface import UserInterface
from lib.ui import App


current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.append(project_path)
from Common.log import user_data_level, user_info_level, user_warning_level


logger.add(os.path.join(project_path, "log/run.log"), rotation="100 MB")

option_dict = SharedMemoryDict(name='option', size=1024)


class User:
    def __init__(self):
        self.power = 'on'
        self.user_option = UserOption()
        self.user_interface = UserInterface()
        self.app = App()
        self.response_time = 0.05
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
        else:
            logger.log("UserWarning", "Option dictionary is broken!")
            
    def run(self):
        t0 = time.time()
        
        self.user_option.update()
        self.user_option.publish()
        
        self.user_interface.update()
        canva_3d = self.user_interface.show()
        
        self.app.show(canva_3d)

        t1 = time.time()
        loop_time = t1 - t0
        logger.log("UserInfo", "Loop time: {}", loop_time)
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)
        

def main():
    logger.log("UserInfo", "User core start.", enqueue=True)
    u = User()
    while True:
        u.run()
        u.update()
        if u.power == 'off':
            logger.log("UserInfo", "User core end.", enqueue=True)
            cv2.destroyAllWindows()
            time.sleep(0.1)
            break

if __name__ == "__main__":
    main()
