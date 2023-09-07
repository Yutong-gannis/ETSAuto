import time
import os
import sys
import math
import scipy
import sympy as sp
import numpy as np

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from lib.changelane import ChangeLane_Helper
from lib.longitude_planner import LongitudePlanner
from lib.planregister import PlanRegister
from lib.objectregister import ObjectRegister
from Message.iodata import load_pkl, save_pkl


class Planner:
    def __init__(self):
        self.desire = 'straight'
        self.plan_register = PlanRegister()
        self.obj_register = ObjectRegister()
        self.long_planner = LongitudePlanner()
        self.changelane_helper = ChangeLane_Helper()
        self.line_m = None
        self.line_l = None
        self.lane_width = 3.6
        self.trajectory = None
        self.condition_dict = None
        
    def update(self):
        lane_dict = load_pkl(os.path.join(project_path, 'Message/temp/line.pkl'))
        condition_dict = load_pkl(os.path.join(project_path, 'Message/temp/condition.pkl'))
        option_dict = load_pkl(os.path.join(project_path, 'Message/temp/option.pkl'))
        dets_dict = load_pkl(os.path.join(project_path, 'Message/temp/dets.pkl'))
        
        if lane_dict is not None:
            self.line_m = lane_dict['line_m']
            self.line_l = lane_dict['line_l']
            self.lane_width = lane_dict['lane_width']
            
        if option_dict is not None:
            self.desire =  option_dict['desire']
            
        if condition_dict is not None:
            self.condition_dict = condition_dict
            
        if dets_dict is not None:
            self.dets_dict = dets_dict
        
            
    def run(self):
        if self.line_m is not None:
            self.trajectory = self.line_m
            self.trajectory, self.plan_register = self.changelane_helper.update(self.trajectory, self.line_l, self.plan_register, self.lane_width, self.desire, self.condition_dict)
        
        leader_position, leader_speed = self.obj_register.update(self.dets_dict, self.trajectory, self.condition_dict, self.lane_width)
        np.save(os.path.join(project_path, 'Message/temp/trajectory.npy'), self.trajectory)
        
        if self.trajectory is not None and len(self.trajectory) >= 50:
            self.trajectory = self.plan_register.update(self.trajectory[:50, :])
            if self.condition_dict is not None:
                plan_speed = self.long_planner.update(self.trajectory, self.condition_dict['speed'])
                save_pkl(os.path.join(project_path, 'Message/temp/planv.pkl'), plan_speed)
