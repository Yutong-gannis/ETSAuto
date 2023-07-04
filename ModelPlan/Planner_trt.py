import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import time
from lib.parse import parse_image


class Supercombo_trt(object):
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

        self.full_img_size = (1164, 874)
        self.input_shape = (512, 256)
        self.parsed_images = []
        self.parsed_images_big = []

        self.pulse_desire_data = np.zeros((1, 100, 8)).astype('float16')  # 意图脉冲数据
        self.desire_data = np.zeros((1, 8)).astype('float16')  # 此时意图
        '''
        0: 直行 1: 左转 2: 右转 3: 左变道 4: 右变道 5: 保持左转 6: 保持右转 7: Null
        '''
        self.traffic_convention_data = np.array([[0, 1]]).astype('float16')
        self.nav_features_data = np.zeros((1, 64)).astype('float16')
        self.initial_state_data = np.zeros((1, 512)).astype('float16')
        self.history_data = np.zeros((1, 99, 128)).astype('float16')

        self.desire_state_start_idx = 5860
        self.desire_state_end_idx = self.desire_state_start_idx + 8
        self.history_start_idx = 5990
        self.history_end_idx = self.history_start_idx + 128  # 记忆特征

    def preprocess(self, img):
        big_img = img
        img = img[360:640, 380:980]
        img = cv2.resize(img, self.input_shape)
        cv2.imshow('input', img)
        big_img = cv2.resize(big_img, self.input_shape)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        img_yuv_big = cv2.cvtColor(big_img, cv2.COLOR_BGR2YUV_I420) / 128.0 - 1.0
        parsed = parse_image(img_yuv)
        parsed_big = parse_image(img_yuv_big)
        if len(self.parsed_images) >= 2:
            del self.parsed_images[0]
            del self.parsed_images_big[0]
        self.parsed_images.append(parsed)
        self.parsed_images_big.append(parsed_big)

    def infer(self, img, desire, nav_feature=None):
        self.preprocess(img)
        if len(self.parsed_images) >= 2:
            parsed_arr = np.array(self.parsed_images)
            parsed_arr.resize((1, 12, 128, 256))
            data = parsed_arr.astype('float16')

            parsed_arr_big = np.array(self.parsed_images_big)
            parsed_arr_big.resize((1, 12, 128, 256))
            data_big = parsed_arr_big.astype('float16')

            self.desire_data = np.zeros((1, 8)).astype('float16')
            self.desire_data[0, desire] = 1

            self.inputs[0]['host'] = np.ravel(data)
            self.inputs[1]['host'] = np.ravel(data_big)
            self.inputs[2]['host'] = np.ravel(self.pulse_desire_data)
            self.inputs[3]['host'] = np.ravel(self.traffic_convention_data)
            self.inputs[4]['host'] = np.ravel(self.nav_features_data)
            self.inputs[5]['host'] = np.ravel(self.history_data)
            for inp in self.inputs:  # 将数据转到gpu
                cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle)  # 推理过程
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)  # 从GPU抓取输出
            self.stream.synchronize()  # 同步视频流
            result = [out['host'] for out in self.outputs][0]
            self.update_history_feature(result)
            self.update_desire(result)
            return result
        return None

    def update_history_feature(self, result):
        feature_i = result[self.history_start_idx:self.history_end_idx]
        for i in range(len(self.history_data[0]) - 1):
            self.history_data[0][i] = self.history_data[0][i + 1]
        self.history_data[0][-1] = feature_i

    def update_desire(self, result):
        desire_pred = result[self.desire_state_start_idx:self.desire_state_end_idx]
        for i in range(len(self.pulse_desire_data[0]) - 1):
            self.pulse_desire_data[0][i] = self.pulse_desire_data[0][i + 1]
        for j in range(7):
            if self.desire_data[0][j] - desire_pred[j] >= 1:
                self.pulse_desire_data[0][-1][j] = self.desire_data[0][j]
            else:
                self.pulse_desire_data[0][-1][j] = 0
            desire_pred[j] = self.desire_data[0][j]
