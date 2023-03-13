import os
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))

import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(project_path, 'Control')))
from controllers.fuzzy_controller import fuzzy_compute


def Follow(cipv, vertical_fuzzy):
    dspeed = int(cipv.speed[1])
    distance = 470 - int(cipv.position[1] + cipv.position[3])
    if -30 < dspeed < 30:
        if 50 < distance < 200:
            acc = fuzzy_compute(vertical_fuzzy, dspeed, distance)
        elif distance >= 200:
            acc = 0.3
        else:
            acc = 1
    elif 30 <= dspeed <= 40:
        acc = 0.7
    elif dspeed > 40:
        acc = 1
    else:
        acc = 0.3
    return acc