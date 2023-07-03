import sys
import os
import numpy as np
import time
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, current_path)
from option import UserOption

user_option = UserOption()
option_list = None
while True:
    user_option.update()
    user_option.publish(project_path)
    time.sleep(0.01)
    try:
        option_list = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
        print(option_list)
    except ValueError:
        print('lost option data')
    if option_list is not None and len(option_list):
        if option_list[2] == 0:
            break
