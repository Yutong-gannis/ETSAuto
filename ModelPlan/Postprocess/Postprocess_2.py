import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from Planprocess import Plans
from Laneprocess import Lanes
from Edgeprocess import Edges
from Leadprocess import Leads
from Poseprocess import Pose


class ModelOutput:
    def __init__(self):
        self.plan_start_idx = 0
        self.plan_end_idx = 4955  # 规划全长4955

        self.lanes_start_idx = self.plan_end_idx
        self.lanes_end_idx = self.lanes_start_idx + 528 + 8  # 车道线全长528 + 车道线置信度 8

        self.road_start_idx = self.lanes_end_idx
        self.road_end_idx = self.road_start_idx + 264  # 路沿 264

        self.lead_start_idx = self.road_end_idx
        self.lead_end_idx = self.lead_start_idx + 102 + 3  # 前车 102 + 0s，2s，4s时刻有leadcar在前方的置信度 3

        self.desire_state_start_idx = self.lead_end_idx
        self.desire_state_end_idx = self.desire_state_start_idx + 8

        self.meta_start_idx = self.desire_state_end_idx
        self.meta_end_idx = self.meta_start_idx + 48

        self.desire_pred_start_idx = self.meta_end_idx
        self.desire_pred_end_idx = self.desire_pred_start_idx + 32

        self.pose_start_idx = self.desire_pred_end_idx
        self.pose_end_idx = self.pose_start_idx + 12  # 自由度数据

        self.recurrent_start_idx = self.pose_end_idx
        self.recurrent_end_idx = self.recurrent_start_idx + 512  # 记忆特征

    def process(self, output):
        if output is None:
            return None, None, None, None, None, None, None, None, None, None
        else:
            plan_line = output[self.plan_start_idx:self.plan_end_idx]
            lanes_line = output[self.lanes_start_idx:self.lanes_end_idx]
            road_edge_line = output[self.road_start_idx:self.road_end_idx]
            lead_line = output[self.lead_start_idx:self.lead_end_idx]
            desire_state_line = output[self.desire_state_start_idx:self.desire_state_end_idx]
            meta_line = output[self.meta_start_idx:self.meta_end_idx]
            desire_pred_line = output[self.desire_pred_start_idx:self.desire_pred_end_idx]
            pose_line = output[self.pose_start_idx:self.pose_end_idx]

            plans = Plans(plan_line)
            lanes_all = Lanes(lanes_line)
            edges_ori = Edges(road_edge_line)
            leads = Leads(lead_line)
            pose = Pose(pose_line)
            plan_position, plan_velocity, plan_acc, plan_yaw, plan_yaw_rate, plan_angle = plans.select()
            lanes = lanes_all.select()
            edges = edges_ori.extract()
            leaders = leads.select()
            return plan_position, plan_velocity, plan_acc, plan_yaw, plan_yaw_rate, plan_angle, lanes, edges, leaders, pose

