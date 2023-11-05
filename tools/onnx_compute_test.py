import onnxruntime
import onnx
import onnx_tool
from onnx2pytorch import ConvertModel
from torchsummary import summary

model_path = r"D:\openpilot\selfdrive\modeld\models\supercombo.onnx"
onnx_model = onnx.load(model_path)
model = ConvertModel(onnx_model, experimental=True).float().cuda()
summary(model, [(12,128,256), (12,128,256)])