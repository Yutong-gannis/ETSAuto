from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import torchvision


class MultiOutputModel1(torch.nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(MultiOutputModel1, self).__init__()
        # self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model = torchvision.models.mobilenet_v3_large(pretrained=True)
        # self.base_model.load_state_dict(torch.load('../input/mobilenetv3/mobilenet_v3_small-047dcff4.pth'))

        last_channel = self.base_model.classifier[-1].out_features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.model1 = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes1),
            nn.ReLU())

        self.model2 = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes2),
            nn.ReLU())

    def forward(self, x):
        x = self.base_model(x)
        # x = self.pool(x)
        output_1 = self.model1(x)
        output_2 = self.model2(x)
        return output_1, output_2


class MultiOutputModel2(torch.nn.Module):
    def __init__(self, num_classes1, num_classes2, num_classes3):
        super(MultiOutputModel2, self).__init__()
        # self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.base_model = torchvision.models.mobilenet_v3_small(pretrained=True)
        # self.base_model.load_state_dict(torch.load('../input/mobilenetv3/mobilenet_v3_small-047dcff4.pth'))

        last_channel = self.base_model.classifier[-1].out_features
        # self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.model1 = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes1),
            nn.ReLU())

        self.model2 = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes2),
            nn.ReLU())

        self.model3 = nn.Sequential(
            nn.Linear(last_channel, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes3),
            nn.ReLU())

    def forward(self, x):
        x = self.base_model(x)
        # x = self.pool(x)
        output_1 = self.model1(x)
        output_2 = self.model2(x)
        output_3 = self.model3(x)
        return output_1, output_2, output_3


