import time
 
import onnx
import numpy as np
import onnxruntime as ort
  
# load onnx model
model = onnx.load_model('alexnet.onnx')
onnx.checker.check_model(model=model)
onnx.helper.printable_graph(graph=model.graph)
    
# onnx_runtime val
# sess_options = ort.SessionOptions()
# sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
# sess_options.optimized_model_filepath = "optimized_mnist_original.onnx"
# session = ort.InferenceSession("mnist_original.onnx", sess_options)
ort_session = ort.InferenceSession("alexnet.onnx")
while 1:
    start = time.time()
    outputs = ort_session.run(None, {'inputs': np.random.randn(10, 3, 224, 224).astype(np.float32)})  # inputs is same to orig
                  
    print(time.time() - start)
