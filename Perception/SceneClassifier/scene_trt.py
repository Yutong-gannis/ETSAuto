import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from albumentations import *
from albumentations.pytorch import ToTensorV2

transforms = {
    x: Compose([
        Resize(416, 416),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.2, border_mode=cv2.BORDER_REPLICATE),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        OneOf([
            GaussianBlur(),
            GaussNoise(),
        ], p=0.2),

        Normalize(),
        ToTensorV2()
    ]) if x == 'train' else Compose([
        Resize(416, 416),

        Normalize(),
        ToTensorV2()
    ]) for x in ['train', 'test']
}


class SceneEngine:
    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs = []
        self.outputs = []
        self.allocations = []
        for i in range(self.engine.num_bindings):
            is_input = False
            if self.engine.binding_is_input(i):
                is_input = True
            name = self.engine.get_binding_name(i)
            dtype = self.engine.get_binding_dtype(i)
            shape = self.engine.get_binding_shape(i)
            if is_input:
                self.batch_size = shape[0]
            size = np.dtype(trt.nptype(dtype)).itemsize
            for s in shape:
                size *= s
            allocation = cuda.mem_alloc(size)
            binding = {
                'index': i,
                'name': name,
                'dtype': np.dtype(trt.nptype(dtype)),
                'shape': list(shape),
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            if self.engine.binding_is_input(i):
                self.inputs.append(binding)
            else:
                self.outputs.append(binding)

    def infer(self, img):
        weathers = ['clear', 'overcast', 'cloudy', 'rainy', 'snowy']
        scenes = ['street', 'highway']
        img = transforms['test'](image=img)['image'].unsqueeze(0).numpy()
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        outputs = []
        for out in self.outputs:
            output = np.zeros(out['shape'], out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            outputs.append(output)
        print(outputs)
        output1 = np.argmax(outputs[0], axis=1)
        output2 = np.argmax(outputs[1], axis=1)
        weather = weathers[output1[0]]
        scene = scenes[output2[0]]
        return weather, scene

'''
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
device = 'cuda:0'
img = cv2.imread(r"D:\Self-driving-Truck-in-Euro-Truck-Simulator2\assets\test3_Moment_small.jpg")
weather_engine_path = r'D:\Self-driving-Truck-in-Euro-Truck-Simulator2\weights\weather_detector.engine'
weather_classifier = SceneEngine(weather_engine_path)
weather, scene = weather_classifier.infer(img)
print(weather, scene)
'''
