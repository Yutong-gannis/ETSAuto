from onnx import load_model, save_model
from onnxmltools.utils import float16_converter

output_onnx_name = '../weights/supercombo_2.onnx'
onnx_model = load_model(output_onnx_name)
trans_model = float16_converter.convert_float_to_float16(onnx_model)
save_model(trans_model, "../weights/supercombo_2_fp16.onnx")