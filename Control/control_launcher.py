import time
import os
import sys
import numpy as np
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from drive import Driver
from Common.constants import T_IDXS

driver = Driver()
control_list = None
option_list = None
speed_list = None
while True:
    try:
        control_list = np.load(os.path.join(project_path, 'temp/control.npy'))
    except ValueError:
        print('lost control data')
    try:
        option_list = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
    except ValueError:
        print('lost option data')
    try:
        speed_list = np.loadtxt(os.path.join(project_path, "temp/speed.txt"), dtype=bytes).astype(float)
    except ValueError:
        print('lost speed data')
    if control_list is not None:
        acc_list = control_list[:10]
        ang_list = control_list[10:20]
        acc_1 = control_list[20]
        ang_1 = control_list[21]
        local_time = time.time()
        abs_time = local_time - control_list[-1]
        reference_index = np.argmin(np.abs(np.array(T_IDXS[:10]) - abs_time))
        acc = 0.5 - acc_list[reference_index] / 2
        ang = 0.5 + ang_list[reference_index] / np.pi / 3
        acc = acc * 0.99 + acc_1 * 0.01
        ang = ang * 0.99 + ang_1 * 0.01
        if speed_list is not None and len(speed_list) == 2:
            if speed_list[1] == 1:
                acc = 0.52
        print('acc: ', acc)
        print('ang: ', ang)
        if option_list is not None and len(option_list):
            if option_list[0] == 0:
                driver.drive(0.5, 0.5)
            elif option_list[0] == 1:
                driver.drive(ang, 0.5)
            elif option_list[0] == 2:
                driver.drive(0.5, acc)
            elif option_list[0] == 3:
                driver.drive(ang, acc)
            if option_list[2] == 0:
                driver.end()
                time.sleep(0.1)
                break
