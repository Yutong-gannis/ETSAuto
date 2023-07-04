import onnxruntime
import numpy as np
import cv2
import time
from lib.parse import parse_image
import win32api


class NAVmodel_onnx(object):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = (256, 256)

    def preprocess(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, self.input_shape)
        return img

    def infer(self, img):
        img = self.preprocess(img)
        img = img.reshape((1, 1, 256, 256)).astype('float32')
        result = self.session.run([self.output_name], {self.input_name: img})
        result = np.array(result)[0]
        nav_feature = result[:, 164:228].astype('float16')
        return nav_feature
