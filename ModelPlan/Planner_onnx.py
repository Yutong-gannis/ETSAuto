import onnxruntime
import numpy as np
import cv2
import time
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
from ModelPlan.lib.parse import parse_image


class Supercombo_onnx_1(object):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        self.imgs_name = self.session.get_inputs()[0].name
        self.desire_name = self.session.get_inputs()[1].name
        self.traffic_convention_name = self.session.get_inputs()[2].name
        self.history_name = self.session.get_inputs()[3].name
        self.output_name = self.session.get_outputs()[0].name

        self.ori_img_shape = [1360, 680]
        self.input_shape = (512, 256)
        self.parsed_images = []

        self.accuracy = 'float32'

        self.pulse_desire_data = np.zeros((1, 8)).astype(self.accuracy)
        self.desire_data = np.zeros((1, 8)).astype(self.accuracy)
        '''
        0: 直行 1: 左转 2: 右转 3: 左变道 4: 右变道 5: 保持左转 6: 保持右转 7: Null
        '''
        self.traffic_convention_data = np.array([[0, 1]]).astype(self.accuracy)
        self.history_data = np.zeros((1, 512)).astype(self.accuracy)

        self.desire_state_start_idx = 5813
        self.desire_state_end_idx = self.desire_state_start_idx + 8
        self.history_start_idx = 5897
        self.history_end_idx = self.history_start_idx + 512

    def preprocess(self, img):
        img = img[360:640, 360:1000]
        img = cv2.resize(img, self.input_shape)
        cv2.imshow('input', img)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        parsed = parse_image(img_yuv)
        if len(self.parsed_images) >= 2:
            del self.parsed_images[0]

        self.parsed_images.append(parsed)

    def infer(self, img, desire):
        self.preprocess(img)
        if len(self.parsed_images) >= 2:
            parsed_arr = np.array(self.parsed_images)
            parsed_arr.resize((1, 12, 128, 256))

            data = parsed_arr.astype(self.accuracy)

            self.desire_data = np.zeros((1, 8)).astype(self.accuracy)
            self.desire_data[0, desire] = 1

            result = self.session.run([self.output_name], {self.imgs_name: data,
                                                           self.desire_name: self.desire_data,
                                                           self.traffic_convention_name: self.traffic_convention_data,
                                                           self.history_name: self.history_data
                                                           })
            result = np.array(result)[0][0]
            self.update_history_feature(result)
            self.update_desire(result)
            return result[:self.history_start_idx]  # 只返回前面的数据
        return None

    def update_desire(self, result):
        desire_pred = result[self.desire_state_start_idx:self.desire_state_end_idx]
        for j in range(7):
            if self.desire_data[0][j] - desire_pred[j] >= 1:
                self.pulse_desire_data[0][j] = self.desire_data[0][j]
            else:
                self.pulse_desire_data[0][j] = 0
            desire_pred[j] = self.desire_data[0][j]

    def update_history_feature(self, result):
        self.history_data = result[self.history_start_idx:self.history_end_idx].reshape((1, 512))


class Supercombo_onnx_2(object):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        self.imgs_name = self.session.get_inputs()[0].name
        self.desire_name = self.session.get_inputs()[1].name
        self.traffic_convention_name = self.session.get_inputs()[2].name
        self.history_name = self.session.get_inputs()[3].name
        self.output_name = self.session.get_outputs()[0].name

        self.input_shape = (512, 256)
        self.parsed_images = []

        self.accuracy = 'float32'

        self.pulse_desire_data = np.zeros((1, 8)).astype(self.accuracy)
        self.desire_data = np.zeros((1, 8)).astype(self.accuracy)
        '''
        0: 直行 1: 左转 2: 右转 3: 左变道 4: 右变道 5: 保持左转 6: 保持右转 7: Null
        '''
        self.traffic_convention_data = np.array([[0, 1]]).astype(self.accuracy)
        self.history_data = np.zeros((1, 512)).astype(self.accuracy)

        self.desire_state_start_idx = 5860
        self.desire_state_end_idx = self.desire_state_start_idx + 8
        self.history_start_idx = 5960
        self.history_end_idx = self.history_start_idx + 512

    def preprocess(self, img):
        img = img[370:620, 380:980]
        img = cv2.resize(img, self.input_shape)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV_I420)
        parsed = parse_image(img_yuv)
        if len(self.parsed_images) >= 2:
            del self.parsed_images[0]

        self.parsed_images.append(parsed)

    def infer(self, img, desire):
        self.preprocess(img)
        if len(self.parsed_images) >= 2:
            parsed_arr = np.array(self.parsed_images)
            parsed_arr.resize((1, 12, 128, 256))

            data = parsed_arr.astype(self.accuracy)

            self.desire_data = np.zeros((1, 8)).astype(self.accuracy)
            self.desire_data[0, desire] = 1

            result = self.session.run([self.output_name], {self.imgs_name: data,
                                                           self.desire_name: self.desire_data,
                                                           self.traffic_convention_name: self.traffic_convention_data,
                                                           self.history_name: self.history_data
                                                           })
            result = np.array(result)[0][0]
            self.update_history_feature(result)
            self.update_desire(result)
            return result[:self.history_start_idx]  # 只返回前面的数据
        return None

    def update_desire(self, result):
        desire_pred = result[self.desire_state_start_idx:self.desire_state_end_idx]
        for j in range(7):
            if self.desire_data[0][j] - desire_pred[j] >= 1:
                self.pulse_desire_data[0][j] = self.desire_data[0][j]
            else:
                self.pulse_desire_data[0][j] = 0
            desire_pred[j] = self.desire_data[0][j]

    def update_history_feature(self, result):
        self.history_data = result[self.history_start_idx:self.history_end_idx].reshape((1, 512))


class Supercombo_onnx_3(object):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        self.imgs_name = self.session.get_inputs()[0].name
        self.big_input_imgs = self.session.get_inputs()[1].name
        self.desire_name = self.session.get_inputs()[2].name
        self.traffic_convention_name = self.session.get_inputs()[3].name
        self.nav_features_name = self.session.get_inputs()[4].name
        self.history_name = self.session.get_inputs()[5].name
        self.output_name = self.session.get_outputs()[0].name

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
        img = img[360:680, 280:1080]
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

            result = self.session.run([self.output_name], {self.imgs_name: data,
                                                           self.big_input_imgs: data_big,
                                                           self.desire_name: self.pulse_desire_data,
                                                           self.traffic_convention_name: self.traffic_convention_data,
                                                           self.nav_features_name: self.nav_features_data,
                                                           self.history_name: self.history_data
                                                           })
            result = np.array(result)[0][0]
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
