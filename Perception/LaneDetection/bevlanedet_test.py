import numpy as np
import cv2
import time
import onnxruntime
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.insert(0, project_path)
from Perception.LaneDetection.lib.postprocess import sigmoid, bev_instance2points_with_offset_z
from Perception.LaneDetection.lib.cluster import embedding_post


class Bev_Lanedet(object):
    def __init__(self, onnx_path):
        self.session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider',
                                                                          'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output0_name = self.session.get_outputs()[0].name
        self.output1_name = self.session.get_outputs()[1].name
        self.output2_name = self.session.get_outputs()[2].name
        self.output3_name = self.session.get_outputs()[3].name

        self.input_shape = (240, 360)
        self.x_range = (3, 53)
        self.y_range = (-6, 6)
        self.meter_per_pixel = 0.5
        self.bev_shape = (
            int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel), int((self.y_range[1] - self.y_range[0]) / self.meter_per_pixel))
        self.trans_image = A.Compose([A.Resize(height=self.input_shape[0], width=self.input_shape[1]),
                                      A.Normalize(),
                                      ToTensorV2()])

        self.post_conf = -0.7  # Minimum confidence on the segmentation map for clustering
        self.post_emb_margin = 6.0  # embeding margin of different clusters
        self.post_min_cluster_size = 20  # The minimum number of points in a cluster

    def preprocess(self, img):
        img = img[50:640, :, :]
        transformed = self.trans_image(image=img)
        cv2.imshow('ori', cv2.resize(img, (720, 480)))
        img = transformed["image"]
        img = img.unsqueeze(0).numpy()
        return img

    def infer(self, img):
        img = self.preprocess(img)
        pred_ = self.session.run([self.output0_name,
                                  self.output1_name,
                                  self.output2_name,
                                  self.output3_name,
                                  ], {self.input_name: img})
        seg = pred_[0]
        print(seg.shape)
        embedding = pred_[1]
        offset_y = sigmoid(pred_[2])
        z_pred = pred_[3]
        
        prediction = (seg, embedding)
        canvas, ids = embedding_post(prediction, self.post_conf, emb_margin=self.post_emb_margin,
                                     min_cluster_size=self.post_min_cluster_size, canvas_color=False)
        offset_y = offset_y[0][0]
        z_pred = z_pred[0][0]
        lines = bev_instance2points_with_offset_z(canvas, max_x=self.x_range[1],
                                                  meter_per_pixal=(self.meter_per_pixel, self.meter_per_pixel),
                                                  offset_y=offset_y, Z=z_pred)
        
        cv2.imshow('canvas', canvas*200)
        
        canva = np.zeros((500, 120))
        pts_x = np.linspace(5, 45, 80)
        '''
        # best_line = lines[np.argmax(line.shape[1] for line in lines)]
        # print(best_line)
        for line in lines:
            fit = np.polyfit(line[0, :], line[1, :], 3)
            pts_y = fit[0] * pts_x ** 3 + fit[1] * pts_x ** 2 + fit[2] * pts_x + fit[3]
            for i in range(len(pts_x)):
                canva = cv2.circle(canva, (60 - int(pts_y[i] * 10), 500 - int(pts_x[i] * 10)), 1, (200, 200, 200), -1)
        '''
        # fit = np.polyfit(best_line[0, :], best_line[1, :], 5)
        # pts_y = fit[0] * pts_x ** 5 + fit[1] * pts_x ** 4 + fit[2] * pts_x ** 3 + fit[3] * pts_x ** 2 + fit[4] * pts_x + fit[5]
        # for i in range(len(pts_x)):
        #     canva = cv2.circle(canva, (40 - int(pts_y[i] * 5), 200 - int(pts_x[i] * 5)), 2, (200, 0, 0), -1)
        return canva


model_path = 'weights/bevlanedet/resnet18_0.5/ep030.onnx'

lane_onnx = Bev_Lanedet(model_path)
cap = cv2.VideoCapture(r"D:\ETSAuto2.0\assets\test.mp4")
while True:
    start = time.time()
    ret, img = cap.read()
    img = lane_onnx.infer(img)
    cv2.imshow('result', img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    end = time.time()
    print('infer time:', str(end - start))
