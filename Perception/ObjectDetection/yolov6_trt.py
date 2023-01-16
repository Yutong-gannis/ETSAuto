from ObjectDetection.utils.utils import BaseEngine

class YOLOPredictor(BaseEngine):
    def __init__(self, engine_path , imgsz=(640,640)):
        super(YOLOPredictor, self).__init__(engine_path)
        self.imgsz = imgsz # your model infer image size
        self.n_classes = 10  # your model classes