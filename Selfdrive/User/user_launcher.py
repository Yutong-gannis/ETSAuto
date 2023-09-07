import sys
import os
import cv2
import numpy as np
import time

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Message.iodata import load_pkl
from lib.option import UserOption
from lib.interface import UserInterface
from lib.ui import App


class User:
    def __init__(self):
        self.power = 'on'
        self.user_option = UserOption()
        self.user_interface = UserInterface()
        
        self.app = App()
        self.response_time = 0.05
        
    def update(self):
        option_dict = load_pkl(os.path.join(project_path, 'Message/temp/option.pkl'))
        if option_dict is not None:
            self.power = option_dict['power']
            
    def run(self):
        t0 = time.time()
        
        self.user_option.update()
        self.user_option.publish()
        
        self.user_interface.update()
        canva_3d = self.user_interface.show()
        
        self.app.show(canva_3d)

        t1 = time.time()
        loop_time = t1 - t0
        if loop_time < self.response_time:
            time.sleep(self.response_time - loop_time)
        

def main():
    time.sleep(5)
    u = User()
    while True:
        u.update()
        u.run()
        if u.power == 'off':
            cv2.destroyAllWindows()
            time.sleep(0.1)
            break

if __name__ == "__main__":
    main()