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
from User.interface import DevInterface
from User.ui import App

root = Tk()
root.geometry("360x360-0+0")
root.title("ETSAuto")
root.iconbitmap(r"D:\ETSAuto4.0\Assets\logo\logo_i.ico")
app = App(root)
app.configure(bg='gray')

user_option = UserOption()
dev_interface = DevInterface()
line_l, line_r, trajectory = None, None, None

while True:
    user_option.update()
    user_option.publish()
    time.sleep(0.1)

    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))
    lane_dict = load_pkl(os.path.join(project_path, 'temp/line.pkl'))
    condition_dict = load_pkl(os.path.join(project_path, 'temp/condition.pkl'))

    try:
        trajectory = np.load(os.path.join(project_path, 'temp/trajectory.npy'), allow_pickle=False)
    except ValueError:
        print('lost trajectory data')

    if lane_dict is not None:
        line_l, line_r = lane_dict['line_l'], lane_dict['line_r']
    user_option.desire = 0
    fps_list = [0.05, 0.05]
    info_canva, scene_canva = dev_interface.show(line_l, line_r, trajectory, option_dict, condition_dict, round(sum(fps_list) / 10, 1))
    line_l, line_r, trajectory = None, None, None
    
    logo = Image.open(r"D:\ETSAuto4.0\Assets\logo\300x300\logo1.png").resize((360, 360))
    logo = ImageTk.PhotoImage(image=logo)
    app.canvas.create_image(180, 180, image=logo)

    info_canva = cv2.cvtColor(info_canva, cv2.COLOR_BGR2RGB)
    info_canva_pil = Image.fromarray(info_canva)
    info_canva_tk = ImageTk.PhotoImage(image=info_canva_pil)
    app.canvas.create_image(140, 180, image=info_canva_tk)

    scene_canva = cv2.cvtColor(scene_canva, cv2.COLOR_BGR2RGB)
    scene_canva_pil = Image.fromarray(scene_canva)
    scene_canva_tk = ImageTk.PhotoImage(image=scene_canva_pil)
    app.canvas.create_image(280, 180, image=scene_canva_tk)

    app.canvas.pack()
    app.update()
    
    if cv2.waitKey(25) & 0xFF == ord('q') or (option_dict is not None and option_dict['power'] == 0):
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break
