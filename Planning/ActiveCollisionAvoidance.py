import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
control_path = os.path.abspath(os.path.join(project_path, 'Control'))
sys.path.insert(0, control_path)
from controllers.purepursuit_controller import Purepursuit

def ActiveCollisionAvoidance(truck, nav_line):
        ang = Purepursuit(truck, nav_line.pts)
        if truck.acc < 0.5:
            truck.acc = 0.5
        else:
            truck.acc = truck.acc + (truck.acc - 0.32) ** 2
            if truck.acc > 1:
                truck.acc = 1
        acc = truck.acc
        return acc, ang