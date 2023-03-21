import torch
import cv2
import os, sys
from albumentations import *
from albumentations.pytorch import ToTensorV2
current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, os.path.abspath(os.path.join(project_path, 'Perception')))
from SceneClassifier.model import MultiOutputModel1

transforms = {
    x: Compose([
        Resize(416, 416),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_REPLICATE),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        OneOf([
            GaussianBlur(),
            GaussNoise(),
        ], p=0.2),

        Normalize(),
        ToTensorV2()
    ]) if x == 'train' else Compose([
        Resize(416, 416),

        Normalize(),
        ToTensorV2()
    ]) for x in ['train', 'test']
}


def load_weather(device):
    path = r"D:\autodrive\Perception\SceneClassifier\weights\weather_scene_mobilenet_v3_412.pt"
    model = MultiOutputModel1(5, 2)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def weather_infer(img, model):
    weathers = ['clear', 'overcast', 'cloudy', 'rainy', 'snowy']
    scenes = ['city street', 'highway', 'residential']
    #timeofdays = ['night', 'daytime', 'dawn/dusk']
    img = transforms['test'](image=img)['image'].unsqueeze(0).to('cuda')
    output1, output2 = model(img)
    output1 = torch.argmax(output1, dim=1).cpu().detach().numpy()
    output2 = torch.argmax(output2, dim=1).cpu().detach().numpy()
    weather = weathers[output1[0]]
    scene = scenes[output2[0]]
    return weather, scene


'''
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda:0'
img = cv2.imread(r"D:\autodrive\assets\test3_Moment_small.jpg")
model = load_weather(device)
output = weather_infer(img, model)
print(output)
'''