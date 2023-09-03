import sys
import os
import cv2
import numpy as np
import time
from PIL import ImageTk
from tkinter import *
import PIL.Image as Image

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Common.iodata import load_pkl
from User.option import UserOption
from User.interface import DevInterface, UserInterface
from User.ui import App

root = Tk()
root.geometry("360x360-0+0")
root.title("ETSAuto")
root.iconbitmap(os.path.join(project_path, "Assets/logo/logo_i.ico"))
app = App(root)
app.configure(bg='gray')

user_option = UserOption()
user_interface = UserInterface()
# dev_interface = DevInterface()
line_l, line_r, trajectory = None, None, None

while True:
    user_option.update()
    user_option.publish()
    time.sleep(0.02)

    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))
    lane_dict = load_pkl(os.path.join(project_path, 'temp/line.pkl'))
    condition_dict = load_pkl(os.path.join(project_path, 'temp/condition.pkl'))

    try:
        trajectory = np.load(os.path.join(project_path, 'temp/trajectory.npy'), allow_pickle=False)
    except ValueError:
        print('lost trajectory data')
    if lane_dict is not None:
        line_l, line_r, line_ll, line_rr, line_z = lane_dict['line_l'], lane_dict['line_r'], lane_dict['line_ll'], lane_dict['line_rr'], lane_dict['line_z']
    fps_list = [0.05, 0.05]
    canva_3d = user_interface.show(line_l, line_r, line_ll, line_rr, trajectory, line_z, option_dict, condition_dict, round(sum(fps_list) / 10, 1))
    # canva_bev = dev_interface.show(line_l, line_r, trajectory, option_dict, condition_dict, round(sum(fps_list) / 10, 1))
    
    logo = Image.open(os.path.join(project_path, "Assets/logo/300x300/logo1.png")).resize((360, 360))
    logo = ImageTk.PhotoImage(image=logo)
    app.canvas.create_image(180, 180, image=logo)
    canva_3d = cv2.cvtColor(canva_3d, cv2.COLOR_BGR2RGB)
    canva_3d_pil = Image.fromarray(canva_3d)
    canva_3d_tk = ImageTk.PhotoImage(image=canva_3d_pil)
    app.canvas.create_image(180, 180, image=canva_3d_tk)
    
    app.canvas.pack()
    app.update()

    # cv2.imshow('bev', canva_bev)
    
    if cv2.waitKey(25) & 0xFF == ord('q') or (option_dict is not None and option_dict['power'] == 0):
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break
