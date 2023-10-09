import os
import numpy as np
import sys
import time
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from controller import Controller


class Control:
    def __init__(self):
        self.power = 'on'
        self.controller = Controller()
        self.response_time = 0.05
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
            
        if 'power' in option_dict_sub.keys():
            self.power = option_dict_sub['power']
            
    def run(self):
        self.controller.update()
        self.controller.run()
        time.sleep(self.response_time)
        
    def exit(self):
        self.controller.exit()


def main():
    time.sleep(10)
    c = Control()
    while True:
        c.update()
        c.run()
        if c.power == 'off':
            c.exit()
            time.sleep(1)
            break

if __name__ == "__main__":
    main()