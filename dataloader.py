from loguru import logger
import sys
import numpy as np
from tqdm import tqdm
import albumentations as A
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os
import cv2
import math
import torch
import time
from torch import multiprocessing


def trans_rotate(pts, rotation):
    T = [[np.cos(rotation), -np.sin(rotation), 0],
        [np.sin(rotation), np.cos(rotation), 0],
        [0, 0, 1]]
    pts_1 = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))[:2, :].T
    return pts_1


def trans_translate(pts, local_pt):
    T = [[1, 0, -local_pt[0]],
         [0, 1, -local_pt[1]],
         [0, 0, 1]]
    pts_1 = np.dot(T, np.concatenate((pts.T, np.ones((1, pts.shape[0]))), axis=0))[:2, :].T
    return pts_1


def world_to_frenet(pts, local_pt, rotation):
    pts_1 = trans_translate(pts, local_pt)
    pts_frenet = trans_rotate(pts_1, rotation)
    return pts_frenet


class ETSMotion(Dataset):
    def __init__(self, dataset_path, normalize=True):
        self.dataset_path = dataset_path
        self.scenes_path = os.listdir(self.dataset_path)
        self.scenes_num = len(self.scenes_path)
        logger.info("{} scenes in the dataset.".format(self.scenes_num))
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])
        self.normalize = normalize

    def __len__(self):
        return len(self.scenes_path)
    
    def __getitem__(self, index):
        video_path = os.path.join(self.dataset_path, self.scenes_path[index], 'video.npz')
        info_path = os.path.join(self.dataset_path, self.scenes_path[index], 'info.npz')
        front = np.load(video_path)['front'].transpose(0,3,2,1)
        leftrear = np.load(video_path)['leftrear'].transpose(0,3,2,1)
        rightrear = np.load(video_path)['rightrear'].transpose(0,3,2,1)
        nav = np.load(video_path)['nav'].transpose(0,3,2,1)
        info = np.load(info_path)['arr_0']
        
        if self.normalize:
            front = np.float16(front) / 255.0
            leftrear = np.float16(leftrear) / 255.0
            rightrear = np.float16(rightrear) / 255.0
            nav = np.float16(nav) / 255.0
        
        hist_trajectory_list, speed_limit_list, stop_list, traffic_convention_list = [], [], [], []
        trajectory_index = np.array([1, 4, 9, 16, 25, 36, 49, 64])
        hist_trajectory_index = np.array([20, 18, 16, 14, 12, 10, 8, 6, 4, 2])
        trajectory_list = []
        video_len = len(info)
        start_index = hist_trajectory_index[0]
        end_index = video_len - trajectory_index[-1]
        sample_len = end_index - start_index
        for i in range(start_index, end_index):
            speed_limit = info[i, 7] / 25
            stop = np.eye(2)[int(info[i, 8])]
            traffic_convention = [0, 1]
            
            current_location = info[i, 0:2]
            current_rotation = info[i, 3]
            hist_trajectory = info[i-hist_trajectory_index, 0:2]
            fut_trajectory = info[trajectory_index+i, 0:2]
            hist_trajectory = world_to_frenet(hist_trajectory, current_location, current_rotation)
            fut_trajectory = world_to_frenet(fut_trajectory, current_location, current_rotation)
            
            speed_limit_list.append(speed_limit)
            stop_list.append(stop)
            traffic_convention_list.append(traffic_convention)
            hist_trajectory_list.append(hist_trajectory)
            trajectory_list.append(fut_trajectory)
        
        front = front[start_index:end_index]
        leftrear = leftrear[start_index:end_index]
        rightrear = rightrear[start_index:end_index]
        nav = nav[start_index:end_index]
        speed_limit_list = np.array(speed_limit_list, dtype=np.float16)
        stop_list = np.array(stop_list, dtype=np.float16)
        traffic_convention_list = np.array(traffic_convention_list, dtype=np.float16)
        hist_trajectory_list = np.array(hist_trajectory_list, dtype=np.float16)
        trajectory_list = np.array(trajectory_list, dtype=np.float16)
        return front, leftrear, rightrear, nav, hist_trajectory_list, speed_limit_list, stop_list, traffic_convention_list, trajectory_list
    
    
if __name__ == '__main__':
    etsmotion = ETSMotion("D:\ETSMotion\ETSMotion")
    train_loader = DataLoader(dataset=etsmotion, batch_size=2, shuffle=False)
    for data in train_loader:
        front_sequence, leftrear_sequence, rightrear_sequence, nav_sequence, hist_trajectory_sequence, speed_limit_sequence, stop_sequence, traffic_convention_sequence, trajectory_sequence = data
    front_sequence = front_sequence.numpy()[1].transpose(0,3,2,1)
    hist_trajectory_sequence = hist_trajectory_sequence.numpy()[1]
    trajectory_sequence = trajectory_sequence.numpy()[1]
    print(front_sequence.shape)
    print(trajectory_sequence.shape)
    print(hist_trajectory_sequence.shape)
    for i in range(len(trajectory_sequence)):
        front = front_sequence[i]
        hist_trajectory = hist_trajectory_sequence[i]
        trajectory = trajectory_sequence[i]
        trajectory_total = np.concatenate((hist_trajectory, np.zeros((1, 2)), trajectory), axis=0) * 4
        img = np.zeros((600, 200, 3), np.uint8)
        cv2.polylines(img, np.int32([trajectory_total + np.array([100, 400])]), False, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (100, 400), 3, (0, 0, 255), -1)
        cv2.imshow('front', np.uint8(front*255))
        cv2.imshow('trajectory', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        