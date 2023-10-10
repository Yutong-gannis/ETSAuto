import win32api
import numpy as np
from loguru import logger
from shared_memory_dict import SharedMemoryDict


class UserOption:
    """Class to record user's option
    """
    def __init__(self):
        self.mode = 'manual'  # 0：手动驾驶 1：激活横向辅助 2：激活纵向辅助 3：激活完全辅助
        self.desire = 'straight'  # 0: 直行 1: 左转 2: 右转 3: 左变道 4: 右变道
        self.power = 'on'  # 0：关闭程序 1：运行程序

    def update_desire(self):  # 更新策略
        if win32api.GetAsyncKeyState(0x60) or win32api.GetAsyncKeyState(0x41) or win32api.GetAsyncKeyState(0x44) or win32api.GetAsyncKeyState(0x53)  or win32api.GetAsyncKeyState(0x57):  # 小键盘0
            self.desire = 'straight'
        elif win32api.GetAsyncKeyState(0x61):  # 小键盘1
            self.desire = 'turnleft'
        elif win32api.GetAsyncKeyState(0x63):  # 小键盘3
            self.desire = 'turnright'
        elif win32api.GetAsyncKeyState(0x64):  # 小键盘4
            self.desire = 'changelaneleft'
        elif win32api.GetAsyncKeyState(0x66):  # 小键盘6
            self.desire = 'changelaneright'

    def update_mode(self):  # 更新模式
        if win32api.GetAsyncKeyState(0x28) or win32api.GetAsyncKeyState(0x41) or win32api.GetAsyncKeyState(0x44) or win32api.GetAsyncKeyState(0x53)  or win32api.GetAsyncKeyState(0x57):  # 小键盘↓键
            self.mode = 'manual'
        elif win32api.GetAsyncKeyState(0x25):  # 小键盘←键
            self.mode = 'latcontrol'
        elif win32api.GetAsyncKeyState(0x27):  # 小键盘→键
            self.mode = 'NAV'
        elif win32api.GetAsyncKeyState(0x26):  # 小键盘↑键
            self.mode = 'AP'

    def update_power(self):
        if win32api.GetAsyncKeyState(0x51) and win32api.GetAsyncKeyState(0x11):  # ctrl+q
            self.power = 'off'

    def update(self):
        self.update_mode()
        self.update_desire()
        self.update_power()
        logger.log("UserData", "Option: {mode}   {desire}   {power}", mode=self.mode, desire=self.desire, power=self.power)
        
        states_dict_sub = SharedMemoryDict(name='states', size=1024)
        if 'lane_change_state' in states_dict_sub.keys():
            if states_dict_sub['lane_change_state'] == 3:
                logger.log("UserInfo", "Lane changing finish, back to go traight.")
                self.desire = 'straight'
        logger.log("UserInfo", "Option uptate finish.")

    def publish(self):
        option_dict_pub = SharedMemoryDict(name='option', size=1024)
        option_dict_pub['mode'] = self.mode
        option_dict_pub['desire'] = self.desire
        option_dict_pub['power'] = self.power
        logger.log("UserInfo", "Option publish finish.")
