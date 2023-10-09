import numpy as np
import win32gui
import win32ui
import win32con
import cv2
import dxcam


class ScreenGraber:
    def __init__(self, region=(0, 40, 1360, 807)):
        """
        This is a class to grab window frame
        
        :param region: The size of window
        :type region: tuple 
        """
        self.region = region
        self.camera = dxcam.create(region=self.region)
    
    def update(self):
        return self.win_grab()
    
    def dxcam_grab(self):
        frame = self.camera.grab()
        return frame
    
    def win_grab(self):
        """ 
        This is update frame function
        
        :return image: Captured frame
        :rtype image: np.array
        """
        hwin = win32gui.GetDesktopWindow()
        left, top, x2, y2 = self.region
        width = x2 - left + 1
        height = y2 - top + 1
        hwindc = win32gui.GetWindowDC(hwin)
        srcdc = win32ui.CreateDCFromHandle(hwindc)
        memdc = srcdc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(srcdc, width, height)
        memdc.SelectObject(bmp)
        memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

        signedIntsArray = bmp.GetBitmapBits(True)
        frame = np.frombuffer(signedIntsArray, dtype='uint8')
        frame.shape = (height, width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        srcdc.DeleteDC()
        memdc.DeleteDC()
        win32gui.ReleaseDC(hwin, hwindc)
        win32gui.DeleteObject(bmp.GetHandle())
        return frame
