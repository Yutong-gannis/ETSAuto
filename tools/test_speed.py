import time
import os
import sys
import torch

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from model import PlanModel
torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    total_time = 0
    batch = 1
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = PlanModel().to(device)
    for i in range(1000):
        t0 = time.time()
        img = torch.zeros((batch, 3, 128, 512)).to(device)
        left_rear_img = torch.zeros((batch, 3, 64, 64)).to(device)
        right_rear_img = torch.zeros((batch, 3, 64, 64)).to(device)
        nav = torch.zeros((batch, 3, 64, 64)).to(device)
        hist_feature = torch.zeros((batch, 40, 128)).to(device)
        actions = torch.zeros((batch, 20, 3)).to(device)
        speed_limit = torch.zeros((batch, 1)).to(device)
        stop = torch.zeros((batch, 2)).to(device)
        traffic_convention = torch.zeros((batch, 2)).to(device)
        
        with torch.no_grad():
            trajectory, buffer = model(img, left_rear_img, right_rear_img, nav, hist_feature, actions, speed_limit, stop, traffic_convention)
            #pool = ThreadPoolExecutor(max_workers=2)
            #thread1 = pool.submit(nav_encoder, nav)
            #thread2 = pool.submit(model, img, left_rear_img, right_rear_img, nav_feature, hist_feature, actions, speed_limit, stop, traffic_convention)
            #nav_feature = thread1.result()
            #trajectory, feature = thread2.result()
            #pool.shutdown()
        t1 = time.time()
        total_time = total_time + t1 - t0
        print('infer:', t1 - t0)
    print('avg time:', total_time/1000)
    