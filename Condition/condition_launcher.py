import os
import sys
import time
import cv2

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from Common.iodata import load_pkl
from Condition.lib.ets2sdktelemetry import Ets2SdkTelemetry
from Condition.lib.sharedmemory import SharedMemory
from Condition.truck_condition import Truck

truck = Truck()
while True:
    option_dict = load_pkl(os.path.join(project_path, 'temp/option.pkl'))
    truck.update()
    truck.publish()
    time.sleep(0.05)

    if option_dict is not None and option_dict['power'] == 0:
        cv2.destroyAllWindows()
        time.sleep(0.1)
        break