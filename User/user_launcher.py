import sys
import os
import cv2
import numpy as np
import time
import pickle

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from User.option import UserOption
from User.interface import DevInterface

user_option = UserOption()
dev_interface = DevInterface()
option_list = None
line_l, line_r, trajectory = None, None, None
condition_list = None

while True:
    user_option.update()
    user_option.publish(project_path)
    time.sleep(0.1)

    try:
        option_list = np.loadtxt(os.path.join(project_path, "temp/option.txt"), dtype=bytes).astype(int)
        print(option_list)
    except ValueError:
        print('lost option data')

    try:
        line_l = np.load(os.path.join(project_path, 'temp/line_l.npy'), allow_pickle=False)
        line_r = np.load(os.path.join(project_path, 'temp/line_r.npy'), allow_pickle=False)
    except ValueError:
        print('lost lane data')

    try:
        trajectory = np.load(os.path.join(project_path, 'temp/trajectory.npy'), allow_pickle=False)
        print(trajectory.shape)
    except ValueError:
        print('lost trajectory data')

    try:
        condition_file = open(os.path.join(project_path, 'temp/condition.pkl'), 'rb')
        condition_list = pickle.load(condition_file)
        condition_file.close()
    except EOFError:
        print('lost condition data')

    user_option.desire = 0
    fps_list = [0.05, 0.05]
    canva = dev_interface.show(line_l, line_r, trajectory, option_list, condition_list, round(sum(fps_list) / 10, 1))
    cv2.imshow('canva', canva)
    line_l, line_r, trajectory = None, None, None

    if cv2.waitKey(25) & 0xFF == ord('q') or option_list[2] == 0:
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break

