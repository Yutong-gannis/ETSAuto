import torch
import torch.nn as nn


class MultipleTrajectoryPredictionLoss(nn.Module):
    def __init__(self, alpha, M, num_pts, distance_type='angle'):
        super().__init__()
        self.alpha = alpha  # TODO: currently no use
        self.M = M
        self.num_pts = num_pts
        
        self.distance_type = distance_type
        if self.distance_type == 'angle':
            self.distance_func = nn.CosineSimilarity(dim=2)
        else:
            raise NotImplementedError
        self.cls_loss = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss(reduction='none')
        # self.reg_loss = SigmoidAbsoluteRelativeErrorLoss()
        # self.reg_loss = AbsoluteRelativeErrorLoss()

    def forward(self, pred_cls, pred_trajectory, gt):
        """
        pred_cls: [B, M]
        pred_trajectory: [B, M * num_pts * 2]
        gt: [B, num_pts, 2]
        """
        assert len(pred_cls) == len(pred_trajectory) == len(gt)
        #pred_trajectory = pred_trajectory.reshape(-1, self.M, self.num_pts, 2)
        with torch.no_grad():
            # step 1: calculate distance between gt and each prediction
            pred_end_positions = pred_trajectory[:, :, self.num_pts-1, :]  # B, M, 2
            gt_end_positions = gt[:, self.num_pts-1:, :].expand(-1, self.M, -1)  # B, 1, 2 -> B, M, 2
            
            distances = 1 - self.distance_func(pred_end_positions, gt_end_positions)  # B, M
            index = distances.argmin(dim=1)  # B

        gt_cls = index
        pred_trajectory = pred_trajectory[torch.tensor(range(len(gt_cls)), device=gt_cls.device), index, ...]  # B, num_pts, 2

        cls_loss = self.cls_loss(pred_cls, gt_cls)

        reg_loss = self.reg_loss(pred_trajectory, gt).mean(dim=(0, 1))

        return cls_loss, reg_loss