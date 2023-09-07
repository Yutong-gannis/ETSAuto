import time
import cv2
import os
import sys
import numpy as np
import onnxruntime

current_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(current_path, '../..'))
sys.path.insert(0, project_path)
sys.path.insert(0, current_path)
from Message.iodata import save_pkl
from lib.utils import xywh2xyxy,  multiclass_nms
from lib.transform import Cam_Transform


class YOLOv8:
    """
    This is class of yolov8 detector

    :param path: The path of yolov8 onnx model
    :type path: str
    """
    def __init__(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=['CUDAExecutionProvider',
                                                                     'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.conf_threshold = 0.5
        self.iou_threshold = 0.5
        self.input_shape = (768, 1360)
        self.size = (640, 640)
        self.classes = ['car', 'bus', 'truck', 'tl_green', 'tl_red', 'tl_yellow']
        self.objects_cls = [0, 1, 2]  # moving objects class ids
        self.tracks_list = [{0: []}, {0: []}, {0: []}]
        self.cam = Cam_Transform()
        self.bev_range = np.array([[0, 50], [-6, 6]])
        self.bev_size = (50, 12)  # range of bev

    def infer(self, img):
        """
        This is a function to do detect

        :param img: The input image
        :type img: np.array
        """
        img = self.preprocess(img)
        outputs = self.session.run([self.output_name], {self.input_name: img})
        objects = self.postprocess(outputs)
        self.publish(objects)

    def preprocess(self, img):
        """
        A function to prepare input data
        
        :param img: The input image
        :return processed_img: prepared data
        :type img: np.array
        :rtype processed_img: np.array
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size[1], self.size[0]))
        img = img / 255.0
        img = img.transpose(2, 0, 1)
        processed_img = np.expand_dims(img, axis=0).astype(np.float32)
        return processed_img

    def postprocess(self, output):
        """
        This is function to do post process
        
        :param output: The output of onnx model
        :return objects_bev: The bev infomation of moving objects
        :type output: np.array
        :rtype objects_bev: np.array 
        """
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        if len(scores) == 0:
            return None
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)
        boxes, scores, class_ids = boxes[indices], scores[indices], class_ids[indices]

        # extract useful information
        objects_image, objects_ids = self.extract_objects(boxes, scores, class_ids)
        objects_bev = self.image_to_bev(objects_image, objects_ids)
        return objects_bev

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.size[1], self.size[0], self.size[1], self.size[0]])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.input_shape[1], self.input_shape[0], self.input_shape[1], self.input_shape[0]])
        return boxes
    
    def image_to_bev(self, objects, objects_ids):
        """"
        THis is function to transform objects from image coordinate into bev coordinate
        
        :param objects: The detect boxes of objects
        :param objects_ids: The ids of objects
        :type objects: np.array
        :type objects_ids: np.array
        :return objects_bev: The bird-eye-view information of objects
        :rtype objects_bev: np.array 
        """
        objects = np.hstack(((objects[:, 2:3] + objects[:, 0:1])/2, objects[:, 3:4]))
        # transform to bev
        objects_bev = self.cam.pixel_to_world(objects)
        # extract objects in bev range
        for i in reversed(range(len(objects_bev))):
            if objects_bev[i, 0] < self.bev_range[0, 0] or objects_bev[i, 0] >= self.bev_range[0, 1] or objects_bev[i, 1] < self.bev_range[1, 0] or objects_bev[i, 1] >= self.bev_range[1, 1]:
                objects_bev = np.delete(objects_bev, i, axis=0)
                objects_ids = np.delete(objects_ids, i, axis=0)
        objects_bev = np.concatenate((objects_bev, objects_ids.reshape(objects_bev.shape[0], 1)), axis=1)
        objects_bev = np.array(objects_bev)
        return objects_bev
    
    def extract_objects(self, boxes, scores, class_ids):
        """
        This is funtion to extract moving objects

        :param boxes: Boxes of all objects
        :param scores: Scores of all objects
        :param class_ids: Class ids of all objects
        :type boxes: np.array
        :type scores: np.array
        :type class_ids: np.array
        :return objects: Boxes of moving objects
        :return objects_ids: Class ids of moving objects
        :rtype objects: np.array
        :rtype objects_ids: np.array
        """
        objects = []
        objects_ids = []
        objects_scores = []
        for i in range(len(class_ids)):
            if class_ids[i] in self.objects_cls:
                objects.append(boxes[i])
                objects_ids.append(class_ids[i])
                objects_scores.append(scores[i])
        objects = np.array(objects)
        objects_ids = np.array(objects_ids)
        return objects, objects_ids
    
    def publish(self, objects):
        """
        This is function to publish data
        
        :param objects: Objects' bev information [x, y, cls]
        :type objects: np.array
        """
        dets_dict = {'objects': objects}
        save_pkl(os.path.join(project_path, 'Message/temp/dets.pkl'), dets_dict)

                 

