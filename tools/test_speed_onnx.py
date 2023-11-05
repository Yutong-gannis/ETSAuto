import time
import os
import sys
import onnxruntime
import numpy as np
from concurrent.futures import ThreadPoolExecutor


if __name__ == '__main__':
    total_time = 0
    batch = 1
    session = onnxruntime.InferenceSession('./planmodel.onnx', providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
    input0_name = session.get_inputs()[0].name
    input1_name = session.get_inputs()[1].name
    input2_name = session.get_inputs()[2].name
    input3_name = session.get_inputs()[3].name
    input4_name = session.get_inputs()[4].name
    input5_name = session.get_inputs()[5].name
    input6_name = session.get_inputs()[6].name
    input7_name = session.get_inputs()[7].name
    input8_name = session.get_inputs()[8].name
    output0_name = session.get_outputs()[0].name
    output1_name = session.get_outputs()[1].name
    
    for i in range(1000):
        t0 = time.time()
        img = np.zeros((batch, 3, 128, 512), dtype=np.float32)
        left_rear_img = np.zeros((batch, 3, 64, 64), dtype=np.float32)
        right_rear_img = np.zeros((batch, 3, 64, 64), dtype=np.float32)
        nav = np.zeros((batch, 3, 64, 64), dtype=np.float32)
        hist_feature = np.zeros((batch, 40, 128), dtype=np.float32)
        actions = np.zeros((batch, 20, 3), dtype=np.float32)
        speed_limit = np.zeros((batch, 1), dtype=np.float32)
        stop = np.zeros((batch, 2), dtype=np.float32)
        traffic_convention = np.zeros((batch, 2), dtype=np.float32)
        
        pred_ = session.run([output0_name,
                             output1_name,],
                            {input0_name: img,
                             input1_name: left_rear_img,
                             input2_name: right_rear_img,
                             input3_name: nav,
                             input4_name: hist_feature,
                             input5_name: actions,
                             input6_name: speed_limit,
                             input7_name: stop,
                             input8_name: traffic_convention,
                             })
        t1 = time.time()
        total_time = total_time + t1 - t0
        print('infer:', t1 - t0)
    print('avg time:', total_time/1000)