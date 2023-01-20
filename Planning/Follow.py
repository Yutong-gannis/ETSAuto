import os
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(project_path, 'Control')))
from controllers.fuzzy_controller import fuzzy_compute


def Follow(cipv, vertical_fuzzy, horizontal_pid, nav_line):
    dspeed = int(cipv.speed[1])
    distance = 470 - int(cipv.position[1] + cipv.position[3])
    #print(dspeed)
    #print(distance)
    horizontal_pid.update_e(float(np.average(nav_line.pts_x[-5:])) - 400)
    acc = fuzzy_compute(vertical_fuzzy, dspeed, distance)
    ang = horizontal_pid.get_u() / np.pi + 0.5
    return acc, ang