import torch
from Perception.LaneDetection.model.single_camera_bev import BEV_LaneDet


model_path = '../weights/bevlanedet/resnet18_0.5/ep020.pth'
input_shape = (240, 360)
output_2d_shape = (144, 256)
x_range = (3, 53)
y_range = (-6, 6)
meter_per_pixel = 0.5
bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel), int((y_range[1] - y_range[0]) / meter_per_pixel))

model = BEV_LaneDet(bev_shape=bev_shape)
pretrained_dict = torch.load(model_path, map_location='cpu')
pretrained_dict1 = pretrained_dict['model_state']
model_dict = model.state_dict()
pretrained_dict = {k[6:]: v for k, v in pretrained_dict1.items()
                   if k[6:] in model_dict.keys()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
model.cuda()
onnx_path = model_path[:-4] + ".onnx"
images = torch.ones((1, 3, 240, 360)).cuda()
with torch.no_grad():
    torch.onnx.export(model, images,
                      onnx_path,
                      verbose=True,
                      opset_version=11,
                      input_names=['input'],
                      output_names=['seg',
                                    'embedding',
                                    'offset_y',
                                    'z_pred'])
    print("Export ONNX successful. Model is saved at", onnx_path)
