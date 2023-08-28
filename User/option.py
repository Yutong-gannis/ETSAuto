import os.path

import win32api
import numpy as np


class UserOption:
    def __init__(self):
        self.mode = 0  # 0：手动驾驶 1：激活横向辅助 2：激活纵向辅助 3：激活完全辅助
        self.desire = 0  # 0: 直行 1: 左转 2: 右转 3: 左变道 4: 右变道
        self.power = 1  # 0：关闭程序 1：运行程序

    def update_desire(self):  # 更新策略
        if win32api.GetAsyncKeyState(0x60):  # 小键盘0
            self.desire = 0
        elif win32api.GetAsyncKeyState(0x61):  # 小键盘1
            self.desire = 1
        elif win32api.GetAsyncKeyState(0x63):  # 小键盘3
            self.desire = 2
        elif win32api.GetAsyncKeyState(0x64):  # 小键盘4
            self.desire = 3
        elif win32api.GetAsyncKeyState(0x66):  # 小键盘6
            self.desire = 4

    def update_mode(self):  # 更新模式
        if win32api.GetAsyncKeyState(0x28):  # 小键盘↓键
            self.mode = 0
        elif win32api.GetAsyncKeyState(0x25):  # 小键盘←键
            self.mode = 1
        elif win32api.GetAsyncKeyState(0x27):  # 小键盘→键
            self.mode = 2
        elif win32api.GetAsyncKeyState(0x26):  # 小键盘↑键
            self.mode = 3

    def update_power(self):
        if win32api.GetAsyncKeyState(0x51) and win32api.GetAsyncKeyState(0x11):  # ctrl+q
            self.power = 0

    def update(self):
        self.update_mode()
        self.update_desire()
        self.update_power()

    def publish(self, project_path):
        publish_data = [self.mode, self.desire, self.power]
        f = open(os.path.join(project_path, "temp/option.txt"), "w")
        for element in publish_data:
            f.write(str(element) + ' ')
        f.close()

