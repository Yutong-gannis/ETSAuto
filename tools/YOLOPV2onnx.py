import onnxruntime
import numpy as np
import cv2

class YOLOPV2onnx():
	def __init__(self, model_path, anchor_path, conf_thres=0.5, iou_thres=0.5):
		# Initialize model
		self.initialize_model(model_path, anchor_path, conf_thres, iou_thres)

	def __call__(self, image):
		return self.estimate_road(image)

	def initialize_model(self, model_path, anchor_path, conf_thres=0.5, iou_thres=0.5):
		self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
		self.conf_thres = conf_thres
		self.iou_thres = iou_thres
		# Read the anchors from the file
		self.anchors = np.squeeze(np.load(anchor_path))
		# Get model info
		self.get_input_details()
		self.get_output_details()

	def estimate_road(self, image):
		input_tensor = self.prepare_input(image)
		# Perform inference on the image
		outputs = self.inference(input_tensor)
		# Process output data
		self.seg_map, self.filtered_boxes, self.filtered_scores = self.process_output(outputs)
		return self.seg_map, self.filtered_boxes, self.filtered_scores

	def prepare_input(self, image):
		self.img_height, self.img_width = image.shape[:2]
		input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		# Resize input image
		input_img = cv2.resize(input_img, (self.input_width,self.input_height))
		# Scale input pixel values to -1 to 1
		mean=[0.485, 0.456, 0.406]
		std=[0.229, 0.224, 0.225]
		input_img = ((input_img/ 255.0 - mean) / std)
		input_img = input_img.transpose(2, 0, 1)
		input_tensor = input_img[np.newaxis,:,:,:].astype(np.float32)
		return input_tensor

	def inference(self, input_tensor):
		# start = time.time()
		outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
		# print(time.time() - start)
		return outputs

	def process_output(self, outputs):
		# Process segmentation map
		seg_map = np.squeeze(np.argmax(outputs[self.output_names.index("segmentation")], axis=1))
		# Process detections
		scores = np.squeeze(outputs[self.output_names.index("classification")])
		boxes = np.squeeze(outputs[self.output_names.index("regression")])
		filtered_boxes, filtered_scores =  self.process_detections(scores, boxes)
		return seg_map, filtered_boxes, filtered_scores

	def process_detections(self, scores, boxes):
		transformed_boxes = transform_boxes(boxes, self.anchors)

		# Filter out low score detections
		filtered_boxes = transformed_boxes[scores>self.conf_thres]
		filtered_scores = scores[scores>self.conf_thres]

		# Resize the boxes with image size
		filtered_boxes[:,[0,2]] *= self.img_width/self.input_width
		filtered_boxes[:,[1,3]] *= self.img_height/self.input_height

		# Perform nms filtering
		filtered_boxes, filtered_scores = nms_fast(filtered_boxes, filtered_scores, self.iou_thres)
		return filtered_boxes, filtered_scores


	def draw_segmentation(self, image, alpha = 0.5):
		return util_draw_seg(self.seg_map, image, alpha)

	def draw_boxes(self, image, text=True):
		return util_draw_detections(self.filtered_boxes, self.filtered_scores, image, text)

	def draw_2D(self, image, alpha = 0.5, text=True):
		front_view = self.draw_segmentation(image, alpha)
		return self.draw_boxes(front_view, text)

	def draw_bird_eye(self, image, horizon_points):
		seg_map = self.draw_2D(image, 0.00001, text=False)
		return util_draw_bird_eye_view(seg_map, horizon_points)

	def draw_all(self, image, horizon_points, alpha = 0.5):
		front_view = self.draw_segmentation(image, alpha)
		front_view = self.draw_boxes(front_view)
		bird_eye_view = self.draw_bird_eye(image, horizon_points)
		combined_img = np.hstack((front_view, bird_eye_view))
		return combined_img

	def get_input_details(self):
		model_inputs = self.session.get_inputs()
		self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]
		self.input_shape = model_inputs[0].shape
		self.input_height = self.input_shape[2]
		self.input_width = self.input_shape[3]

	def get_output_details(self):

		model_outputs = self.session.get_outputs()
		self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]