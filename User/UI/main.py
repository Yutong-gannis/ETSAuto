from tkinter import *
from PIL import Image, ImageTk

window = Tk()
window.title("ESTAuto")
canva = Canvas(window, width=400, height=300, bg="gray")
im = Image.fromarray(img)
imgtk = ImageTk.PhotoImage(image=im)
Label(window, image= imgtk).pack()
canva.pack()
window.mainloop()
