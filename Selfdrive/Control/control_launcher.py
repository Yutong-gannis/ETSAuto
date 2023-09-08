import os
import numpy as np
import sys
import time

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Message.iodata import load_pkl, save_pkl
from controller import Controller


class Control:
    def __init__(self):
        self.power = 'on'
        self.controller = Controller()
        self.response_time = 0.05
        
    def update(self):
        option_dict = load_pkl(os.path.join(project_path, 'Message/temp/option.pkl'))
        if option_dict is not None:
            self.power = option_dict['power']
            
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