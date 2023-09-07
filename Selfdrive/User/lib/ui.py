import cv2
from PIL import ImageTk
from tkinter import *
import PIL.Image as Image
from ttkbootstrap.constants import *
from ttkbootstrap import Style
from tkinter import *
import PIL.Image as Image
import time
import sys
import os
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../../..'))

class App(Frame):
    """Class of tkinter app

    :param Frame: 
    :type Frame:
    """
    def __init__(self, master=Tk()):
        super().__init__(master)
        self.master = master
        self.master.geometry("360x360-0+0")
        self.master.title("ETSAuto")
        self.master.iconbitmap(os.path.join(project_path, "Assets/logo/logo_i.ico"))
        self.pack(side='left', anchor='nw')
        self.style = Style(theme='journal')
        self.window = self.style.master
        self.createwedget()
        self.configure(bg='gray')

    def createwedget(self):
        self.canvas=Canvas(self, width=360,height=360, bg="black")
        self.canvas.pack()
        logo = Image.open(os.path.join(project_path, "Assets/logo/300x300/logo1.png")).resize((360, 360))
        logo = ImageTk.PhotoImage(image=logo)
        self.canvas.create_image(180,180,image=logo)
        self.update()
        time.sleep(3)

        # self.btn01=ttk.Button(self.window, text='dev', bootstyle=(INFO, OUTLINE))
        # self.btn01.place(x=5, y=30, width=60, height=40)

    def show(self, canva):
        canva = cv2.cvtColor(canva, cv2.COLOR_BGR2RGB)
        canva_pil = Image.fromarray(canva)
        canva_tk = ImageTk.PhotoImage(image=canva_pil)
        self.canvas.create_image(180, 180, image=canva_tk)
        
        self.canvas.pack()
        self.update()