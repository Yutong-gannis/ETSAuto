import numpy as np
import scipy
from loguru import logger


class LongitudePlanner:
    def __init__(self):
        self.speedlimit = 0
        self.current_speed = 0
        self.min_speed = 3
        self.max_accelerate = 3
    
    def update(self, current_speed, speedlimit):
        self.current_speed = current_speed
        self.speedlimit = speedlimit
    
    def run(self, trajectory):
        v_plan = self.plan_v_by_curvature(trajectory)
        logger.log("PlanningInfo", "Longitude Planning finish")
        return v_plan
    
    def get_arc_curve(self, pts):
        x_t = np.gradient(pts[:, 0])
        y_t = np.gradient(pts[:, 1])
        xx_t = np.gradient(x_t)
        yy_t = np.gradient(y_t)
        curvature = np.average(np.abs(xx_t * y_t - x_t * yy_t) / (x_t * x_t + y_t * y_t) ** 1.5)
        curvature = 1 / curvature
        return curvature
        
    def plan_v_by_curvature(self, lane_m):
        lane_curve = self.get_arc_curve(lane_m)
        max_curvature = 650
        min_curvature = 0
        if lane_curve < min_curvature:
            lane_curve = min_curvature
        elif lane_curve > max_curvature:
            lane_curve = max_curvature
        
        normalized_curvature = (lane_curve - min_curvature) / (max_curvature - min_curvature)

        decel = (self.speedlimit - self.min_speed) * (1 - normalized_curvature)
        v_plan = self.speedlimit  - decel
        
        if lane_curve < max_curvature:
            if self.current_speed - v_plan > self.max_accelerate/20:  # limit deceleration
                v_plan = self.current_speed - self.max_accelerate/20
            elif self.current_speed > self.min_speed and v_plan - self.current_speed > self.max_accelerate/20:  # acceleration
                v_plan = self.current_speed + self.max_accelerate/20
        if v_plan > self.speedlimit:
            v_plan = self.speedlimit

        return v_plan

    