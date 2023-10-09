import time
import os
import sys
import math
import scipy
import sympy as sp
import numpy as np
from shared_memory_dict import SharedMemoryDict

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from lib.changelane import ChangeLane_Helper
from lib.longitude_planner import LongitudePlanner
from lib.planregister import PlanRegister
from lib.objectregister import ObjectRegister


class Planner:
    def __init__(self):
        self.desire = 'straight'
        self.mode = 'manual'
        self.plan_register = PlanRegister()
        self.obj_register = ObjectRegister()
        self.long_planner = LongitudePlanner()
        self.changelane_helper = ChangeLane_Helper()
        self.line_m = None
        self.line_l = None
        self.lane_width = 3.6
        self.trajectory = None
        self.nav_line = None
        self.condition_dict = None
        self.dets_dict = None
        
    def update(self):
        option_dict_sub = SharedMemoryDict(name='option', size=1024)
        lane_dict_sub = SharedMemoryDict(name='lane', size=1024)
        condition_dict_sub = SharedMemoryDict(name='condition', size=1024)
        dets_dict_sub = SharedMemoryDict(name='dets', size=1024)
        nav_dict_sub = SharedMemoryDict(name='nav', size=1024)
        
        if "line_m" in lane_dict_sub.keys():
            self.line_m = lane_dict_sub['line_m']
            self.line_l = lane_dict_sub['line_l']
            self.lane_width = lane_dict_sub['lane_width']
            
        if 'desire' in option_dict_sub.keys():
            self.desire =  option_dict_sub['desire']
            self.mode = option_dict_sub['mode']
            
        
        if 'nav_line' in nav_dict_sub.keys():
            self.nav_line = nav_dict_sub['nav_line']
        
        if len(condition_dict_sub):
            self.condition_dict = condition_dict_sub
            
        if len(dets_dict_sub):
            self.dets_dict = dets_dict_sub
        
            
    def run(self):
        self.trajectory = self.line_m
        self.trajectory, self.plan_register = self.changelane_helper.update(self.trajectory, self.line_l, self.plan_register, self.lane_width, self.desire, self.condition_dict)
        
        if self.mode == 'NAV' and self.nav_line is not None:
            self.trajectory = self.nav_line
        
        leader_position, leader_speed = self.obj_register.update(self.dets_dict, self.trajectory, self.condition_dict, self.lane_width)
        plan_dict_pub = SharedMemoryDict(name='plan', size=1024)
        plan_dict_pub['trajectory'] = self.trajectory
        
        if self.trajectory is not None and len(self.trajectory) >= 50:
            self.trajectory = self.plan_register.update(self.trajectory[:50, :])
            print(self.trajectory.shape)
            if self.condition_dict is not None:
                self.long_planner.update(self.condition_dict['speed'], self.condition_dict['speedlimit'])
                plan_speed = self.long_planner.run(self.trajectory[:30, :])
                plan_dict_pub['planv'] = plan_speed
