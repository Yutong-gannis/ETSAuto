import cv2
from PIL import ImageTk
from tkinter import *
from tkinter import ttk
import PIL.Image as Image
from ttkbootstrap.constants import *
from ttkbootstrap import Style
import time
import sys
import os
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)

class App(Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack(side='left', anchor='nw')
        self.style = Style(theme='journal')
        self.window = self.style.master
        self.createwedget()

    def createwedget(self):
        self.canvas=Canvas(self, width=360,height=360, bg="black")
        self.canvas.pack()
        logo = Image.open(r"D:\ETSAuto4.0\Assets\logo\300x300\logo1.png").resize((360, 360))
        logo = ImageTk.PhotoImage(image=logo)
        self.canvas.create_image(180,180,image=logo)
        self.update()
        time.sleep(5)

        self.btn01=ttk.Button(self.window, text='dev', bootstyle=(INFO, OUTLINE))
        self.btn01.place(x=5, y=30, width=60, height=40)

