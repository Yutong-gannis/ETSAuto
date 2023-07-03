from ModelPlan.Planner_onnx import Supercombo_onnx_1, Supercombo_onnx_2, Supercombo_onnx_3
from ModelPlan.Planner_trt import Supercombo_trt_1, Supercombo_trt_2, Supercombo_trt_3


def load_model(supercombo_path, version, engine):
    supercombo = eval('Supercombo_' + engine + '_' + str(version))(supercombo_path)
    return supercombo
