import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.mobilenetv3 import mobilenet_v3_large, mobilenet_v3_small
from efficientnet_pytorch import EfficientNet


class PlanModel(nn.Module):
    def __init__(self, num_cls=5, num_pts=8):
        super().__init__()
        self.num_cls = num_cls
        self.num_pts = num_pts
        self.imgencoder = ImgEncoder()
        self.leftrearencoder = RearEncoder()
        self.rightrearencoder = RearEncoder()
        self.navencoder = NAVEncoder()
        self.histencoder = HistEncoder()
        self.actionencoder = ActionEncoder()
        self.backbone = Backbone()
        self.neck = Neck()
        self.fc = nn.Sequential(nn.Linear(640, 128),
                                nn.ReLU())

    def forward(self, img, left_rear_img, right_rear_img, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention):
        img_feature = self.imgencoder(img)  # batch x 512
        left_rear_feature = self.leftrearencoder(left_rear_img)  # batch x 64
        right_rear_feature = self.rightrearencoder(right_rear_img)  # batch x 64
        frame_feature = torch.cat((img_feature, left_rear_feature, right_rear_feature), dim=1) # batch x 640
        
        hist_feature = self.histencoder(hist_feature)  # batch x 1024
        nav_feature = self.navencoder(nav, speed_limit, stop, traffic_convention)  # batch x 256
        hist_trajectory_feature = self.actionencoder(hist_trajectory)  # batch x 128
        feature = torch.cat((frame_feature, hist_feature, nav_feature, hist_trajectory_feature), dim=1)  # batch x 2048
        
        feature = self.backbone(feature) # batch x 256
        #action = self.neck(feature)  # batch x 2
        pred = self.neck(feature)  # batch x (5 x 8 x 2 + 5)
        pred_cls = pred[:, :self.num_cls]
        pred_trajectory = pred[:, self.num_cls:].reshape(-1, self.num_cls, self.num_pts, 2)

        #pred_xs = pred_trajectory[:, :, :, 0:1].exp()
        #pred_ys = pred_trajectory[:, :, :, 1:2].sinh()
        #pred_zs = pred_trajectory[:, :, :, 2:3]
        #pred_trajectory, torch.cat((pred_xs, pred_ys, pred_zs), dim=3)
        
        buffer = self.fc(frame_feature).reshape((-1, 1, 128))
        # print("model output:" + str(action.shape))
        return pred_cls, pred_trajectory, buffer


class ImgEncoder(nn.Module):
    def __init__(self):
        super(ImgEncoder, self).__init__()
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        #input_dim = 1408
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b1', in_channels=3)
        #input_dim = 1280
        self.backbone = mobilenet_v3_large(pretrained=True)
        input_dim = 960
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, 32, 1),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.ELU())
        self.resblock = ResBlock(2048, 512, 512)
        
    def forward(self, x):
        #feature = self.backbone.extract_features(x)
        feature = self.backbone.features(x)
        feature = self.conv(feature)
        feature = self.resblock(feature)
        # print("backbone output:" + str(feature.shape))
        return feature  # batch x 512
    

class RearEncoder(nn.Module):
    def __init__(self):
        super(RearEncoder, self).__init__()
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        #input_dim = 1408
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b0', in_channels=3)
        #input_dim = 1280
        #self.backbone = mobilenet_v3_large(pretrained=True)
        #input_dim = 960
        self.backbone = mobilenet_v3_small(pretrained=True)
        input_dim = 576
        self.conv = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, 32, 1),
            nn.BatchNorm2d(32),
            nn.Flatten(),
            nn.ELU())
        self.resblock = ResBlock(32*4, 512, 64)
        
    def forward(self, x):
        #feature = self.backbone.extract_features(x)
        feature = self.backbone.features(x)
        #print(feature.shape)
        feature = self.conv(feature)
        #print(feature.shape)
        feature = self.resblock(feature)
        # print("backbone output:" + str(feature.shape))
        return feature  # batch x 64


class NAVEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #self.backbone = EfficientNet.from_pretrained('efficientnet-b2', in_channels=3)
        #input_dim = 1408
        #self.backbone = mobilenet_v3_large(pretrained=True)
        #input_dim = 960
        self.backbone = mobilenet_v3_small(pretrained=True)
        input_dim = 576
        self.nav_head = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.Conv2d(input_dim, 24, 1),
            nn.BatchNorm2d(24),
            nn.Flatten(),
            nn.Linear(96, 256),
            nn.ELU(),
        )
        self.resblock_1 = ResBlock(256, 512, 256)
        self.resblock_2 = ResBlock(256, 512, 256)
        self.fc1 = nn.Sequential(nn.Linear(256, 256),
                                 nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(256+1+2+2, 256),
                                 nn.ReLU())
        
    def forward(self, nav, speed_limit, stop, traffic_convention):
        #feature = self.backbone.extract_features(x)
        feature = self.backbone.features(nav)
        feature = self.nav_head(feature)
        #print(feature.shape)
        feature = self.resblock_1(feature)
        feature = self.resblock_2(feature)
        feature = self.fc1(feature)
        feature = torch.cat((feature, speed_limit, stop, traffic_convention), dim=1)
        out = self.fc2(feature)
        return out
    

class ActionEncoder(nn.Module):
    def __init__(self):
        super(ActionEncoder, self).__init__()
        self.flatten = nn.Flatten()
        self.resblock_1 = ResBlock(20, 512, 512)
        self.resblock_2 = ResBlock(512, 512, 128)
        
    def forward(self, action):
        feature = self.flatten(action)
        feature = self.resblock_1(feature)
        feature = self.resblock_2(feature)
        return feature


class HistEncoder(nn.Module):
    def __init__(self):
        super(HistEncoder, self).__init__()
        self.index = [-5, -10, -15, -20, -25, -30, -35, -40]
        self.flatten = nn.Flatten()
        self.resblock = ResBlock(1024, 1024, 1024)
        self.fc = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        
    def forward(self, hist_feature):
        feature = hist_feature[:, self.index, :]
        feature = self.flatten(feature)
        feature = self.resblock(feature)
        feature = self.relu(self.fc(feature))
        return feature


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU())
        self.resblock_1 = ResBlock(512, 1024, 512)
        self.resblock_2 = ResBlock(512, 1024, 512)
        self.resblock_3 = ResBlock(512, 1024, 512)
        
    def forward(self, feature):
        feature = self.fc(feature)
        feature = self.resblock_1(feature)
        feature = self.resblock_2(feature)
        feature = self.resblock_3(feature)
        return feature
    
    
class Neck(nn.Module):
    def __init__(self):
        super(Neck, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(512, 256),
                                   nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 5*(8*2+1)),
                                   nn.ReLU())
        self.resblock = ResBlock(256, 256, 128)
        
    def forward(self, x):
        feature = self.fc1(x)
        feature = self.resblock(feature)
        feature = self.fc2(feature)
        return feature
        

class ResBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        identity = x
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x += identity
        x = self.relu(x)
        return x
    

if __name__ == '__main__':
    model = PlanModel().cuda().half()
    
    batch = 5
    img = torch.zeros((batch, 3, 128, 512)).cuda().half()
    left_rear_img = torch.zeros((batch, 3, 64, 64)).cuda().half()
    right_rear_img = torch.zeros((batch, 3, 64, 64)).cuda().half()
    nav = torch.zeros((batch, 3, 64, 64)).cuda().half()
    hist_feature = torch.zeros((batch, 49, 128)).cuda().half()
    hist_trajectory = torch.zeros((batch, 10, 2)).cuda().half()
    speed_limit = torch.zeros((batch, 1)).cuda().half()
    stop = torch.zeros((batch, 2)).cuda().half()
    traffic_convention = torch.zeros((batch, 2)).cuda().half()
    cls, trajectory, feature = model(img, left_rear_img, right_rear_img, nav, hist_feature, hist_trajectory, speed_limit, stop, traffic_convention)
    print("pred shape:" + str(trajectory.shape))