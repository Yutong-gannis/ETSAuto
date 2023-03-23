import os
import sys
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
from postprocess import postprocess  # 后处理


class YOLOPredictor(object):
    def __init__(self, engine_path, imgsz=(640, 640)):
        self.imgsz = imgsz
        self.n_classes = 10
        self.class_names = ["person", "rider", "car", "bus", "truck", "tl_green", "tl_red", "tl_yellow", "tl_none",
                            "traffic sign"]
        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # initialize TensorRT plugins
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

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        for inp in self.inputs:  # 将数据转到gpu
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle)  # 推理过程
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)  # 从GPU抓取输出
        self.stream.synchronize()  # 同步视频流
        data = [out['host'] for out in self.outputs]
        return data

    def inference(self, origin_img, CAM, CAM_BL, CAM_BR, img_show, tracker, tracks, infer_time, conf=0.5):
        img, ratio = preproc(origin_img, self.imgsz)
        data = self.infer(img)
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                               np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        objs = None
        obstacles = None
        traffic_light = None
        if dets is not None and len(dets):
            obstacles, objs, tracks, traffic_light = postprocess(dets, CAM, CAM_BL, CAM_BR, tracker, tracks, infer_time)
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            img_show = vis(img_show, final_boxes, final_scores, final_cls_inds, conf=conf,
                            class_names=self.class_names)
        return img_show, obstacles, objs, tracks, traffic_light


def preproc(img, input_size, swap=(2, 0, 1)):
    padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        '''
        #text = '{}'.format(class_names[cls_id])
        #txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        #txt_size = cv2.getTextSize(text, font, 1, 2)[0]
        #txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        #cv2.rectangle(img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), txt_bk_color, -1)
        #cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 1, txt_color, thickness=2)
        '''
    return img


_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
    ]
).astype(np.float32).reshape(-1, 3)
