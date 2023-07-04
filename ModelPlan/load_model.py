from ModelPlan.Planner_onnx import Supercombo_onnx
from ModelPlan.Planner_trt import Supercombo_trt


def load_model(supercombo_path, engine):
    supercombo = eval('Supercombo_' + engine)(supercombo_path)
    return supercombo
