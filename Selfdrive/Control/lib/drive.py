import pyvjoy


class Driver:
    def __init__(self):
        self.joy = pyvjoy.VJoyDevice(1)

    def drive(self, ang, acc):
        self.joy.data.wAxisX = int(ang * 32767)
        self.joy.data.wAxisY = int(acc * 32767)
        self.joy.data.wAxisZ = 0
        self.joy.update()

    def end(self):
        self.joy.data.wAxisX = int(0.5 * 32767)
        self.joy.data.wAxisY = int(0.5 * 32767)
        self.joy.data.wAxisZ = 0
        self.joy.update()
