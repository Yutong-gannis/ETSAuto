import os

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

import numpy as np
import sys

sys.path.insert(0, project_path)
from Control.controllers.purepursuit_controller import Purepursuit


def Cruise(vertical_pid, horizontal_pid, truck, nav_line):
    vertical_pid.update_e(truck['speedlimit'] - truck['speed'])
    acc = vertical_pid.get_a()
    # horizontal_pid.update_e(np.average(nav_line[0:5], axis=0)[1] - 0)
    # ang = horizontal_pid.get_u() / np.pi + resnet34_0.5
    ang = Purepursuit(truck, nav_line)

    if acc > 0.5:
        acc = 0.5
    elif acc < -0.5:
        acc = -0.5
    return acc, ang
