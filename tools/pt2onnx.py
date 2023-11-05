import torch
import os
import sys
import onnx
from onnx import load_model, save_model
import onnxoptimizer
from onnxmltools.utils import float16_converter
from onnxsim import simplify

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '..'))
sys.path.insert(0, project_path)
from model import PlanModel


batch = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = PlanModel().to(device)
onnx_path = "planmodel.onnx"
img = torch.zeros((batch, 3, 128, 512)).to(device)
left_rear_img = torch.zeros((batch, 3, 64, 64)).to(device)
right_rear_img = torch.zeros((batch, 3, 64, 64)).to(device)
nav = torch.zeros((batch, 3, 64, 64)).to(device)
hist_feature = torch.zeros((batch, 40, 128)).to(device)
actions = torch.zeros((batch, 20, 3)).to(device)
speed_limit = torch.zeros((batch, 1)).to(device)
stop = torch.zeros((batch, 2)).to(device)
traffic_convention = torch.zeros((batch, 2)).to(device)
with torch.no_grad():
    torch.onnx.export(model, (img, left_rear_img, right_rear_img, nav, hist_feature, actions, speed_limit, stop, traffic_convention),
                      onnx_path,
                      verbose=False,
                      opset_version=11,
                      do_constant_folding=False,
                      input_names=['front', 
                                   'leftrear', 
                                   'rightrear', 
                                   'nav', 
                                   'features_buffer', 
                                   'hist_trajectory', 
                                   'speed_limit', 
                                   'stop', 
                                   'traffic_convention'],
                      output_names=['trajectory',
                                    'feature'])
    print("Export ONNX successful. Model is saved at", onnx_path)
    
'''
onnx_model = load_model(onnx_path)
trans_model = float16_converter.convert_float_to_float16(onnx_model, keep_io_types=True)
save_model(trans_model, fp16_onnx_path)

model = onnx.load(sim_onnx_path)
model = onnxoptimizer.optimize(model)
onnx.save(model, sim_onnx_path)
'''
