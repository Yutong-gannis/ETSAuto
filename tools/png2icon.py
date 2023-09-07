import os
from PIL import Image


file = r"D:\ETSAuto4.0\Assets\logo\logo_i.png"
size = (24,24)

tmp = os.path.splitext(file)
if tmp[1] == '.png':
    outName = tmp[0] + '.ico'
    # 打开图片并设置大小
    im = Image.open(file).resize(size)
    try:
        # 图标文件保存至icon目录
        path = os.path.join('icon', outName)
        im.save(path)

        print('{} --> {}'.format(file, outName))
    except IOError:
        print('connot convert :',file)