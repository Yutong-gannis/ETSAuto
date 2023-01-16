import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.interpolate import InterpolatedUnivariateSpline
from .postprocess import postprocess, middel_line, line_classification

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255)
]


class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = InterpolatedUnivariateSpline(points[:, 1],
                                                     points[:, 0],
                                                     k=min(3,
                                                           len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.metadata = metadata or {}

        self.sample_y = range(710, 150, -10)
        self.ori_img_w = 1280
        self.ori_img_h = 720

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)

        lane_xs[(lane_ys < self.min_y) |
                (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self):
        sample_y = self.sample_y
        img_w, img_h = self.ori_img_w, self.ori_img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)
        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                              axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration

class CLRNet:
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

            self.conf_threshold = 0.6
        self.nms_thres = 90
        self.max_lanes = 4
        self.sample_points = 48
        self.num_points = 72
        self.n_offsets = 72
        self.n_strips = 71
        self.img_w = 1280
        self.img_h = 720
        self.ori_img_w = 1280
        self.ori_img_h = 720
        self.cut_height = 320

        self.input_width = 800
        self.input_height = 320

        self.sample_x_indexs = (np.linspace(0, 1, self.sample_points) * self.n_strips)
        self.prior_feat_ys = np.flip((1 - self.sample_x_indexs / self.n_strips))
        self.prior_ys = np.linspace(1,0, self.n_offsets)
    
    def softmax(self, x, axis=None):
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)

    def Lane_nms(self, proposals,scores,overlap=50, top_k=4):
        keep_index = []
        sorted_score = np.sort(scores)[-1] # from big to small 
        indices = np.argsort(-scores) # from big to small 
        
        r_filters = np.zeros(len(scores))

        for i, indice in enumerate(indices):
            if r_filters[i] == 1:  # continue if this proposal is filted by nms before
                continue
            keep_index.append(indice)
            if len(keep_index)>top_k:  # break if more than top_k
                break
            if i == (len(scores)-1):  # break if indice is the last one
                break
            sub_indices = indices[i+1:]
            for sub_i, sub_indice in enumerate(sub_indices):
                r_filter = self.Lane_IOU(proposals[indice,:],proposals[sub_indice,:],overlap)
                if r_filter: r_filters[i+1+sub_i]=1 
        num_to_keep = len(keep_index)
        keep_index = list(map(lambda x: x.item(), keep_index))
        return keep_index, num_to_keep
    
    def Lane_IOU(self, parent_box, compared_box, threshold):
        '''
        calculate distance one pair of proposal lines
        return True if distance less than threshold 
        '''
        n_offsets = 72
        n_strips = n_offsets - 1

        start_a = (parent_box[2] * n_strips + 0.5).astype(int) # add 0.5 trick to make int() like round  
        start_b = (compared_box[2] * n_strips + 0.5).astype(int)
        start = max(start_a,start_b)
        end_a = start_a + parent_box[4] - 1 + 0.5 - (((parent_box[4] - 1)<0).astype(int))
        end_b = start_b + compared_box[4] - 1 + 0.5 - (((compared_box[4] - 1)<0).astype(int))
        end = min(min(end_a,end_b),71)
        if (end - start)<0:
            return False
        dist = 0
        for i in range(5+start,5 + int(end)):
            if i>(5+end):
                 break
            if parent_box[i] < compared_box[i]:
                dist += compared_box[i] - parent_box[i]
            else:
                dist += parent_box[i] - compared_box[i]
        return dist < (threshold * (end - start + 1))

    def predictions_to_pred(self, predictions):
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(self.prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            mask = ~((((lane_xs[:start] >= 0.) & (lane_xs[:start] <= 1.)
                       )[::-1].cumprod()[::-1]).astype(np.bool))

            lane_xs[end + 1:] = -2
            lane_xs[:start][mask] = -2
            lane_ys = self.prior_ys[lane_xs >= 0]
            lane_xs = lane_xs[lane_xs >= 0]

            lane_xs = np.double(lane_xs)
            lane_xs = np.flip(lane_xs, axis=0)
            lane_ys = np.flip(lane_ys, axis=0)
            lane_ys = (lane_ys * (self.ori_img_h - self.cut_height) +
                       self.cut_height) / self.ori_img_h
            if len(lane_xs) <= 1:
                continue

            points = np.stack(
                (lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)),
                axis=1).squeeze(2)

            lane = Lane(points=points,
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            scores = self.softmax(predictions[:, :2], 1)[:, 1]

            keep_inds = scores >= self.conf_threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue
            nms_predictions = predictions

            nms_predictions = np.concatenate(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], axis=-1)
    
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[...,5:] = nms_predictions[..., 5:] * (self.img_w - 1)
            
            keep, num_to_keep = self.Lane_nms(
                nms_predictions,
                scores,
                self.nms_thres,
                self.max_lanes)

            keep = keep[:num_to_keep]
            predictions = predictions[keep]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = np.round(predictions[:, 5] * self.n_strips)
            pred = self.predictions_to_pred(predictions)
            decoded.append(pred)
            
        return decoded


    def forward(self, img, im1, CAM):
        img = img[self.cut_height:, :, :]
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(np.float32(img[:, :, :, np.newaxis]), (3, 2, 0, 1))
        img = np.ascontiguousarray(img)
        cuda.memcpy_htod(self.inputs[0]['allocation'], img)
        self.context.execute_v2(self.allocations)
        outputs = []
        for out in self.outputs:
            output = np.zeros(out['shape'],out['dtype'])
            cuda.memcpy_dtoh(output, out['allocation'])
            outputs.append(output)
        output = outputs[0]
        output = self.get_lanes(output)
        res, bev_lanes = postprocess(im1, output, CAM)
        return res, bev_lanes








'''
# 测试程序
if __name__ == "__main__":
    cap = cv2.VideoCapture("D:/autodrive/script/test3.mp4")
    isnet = CLRNet('tusimple_r18.engine', np.ones(shape=(3,3)))
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('D:/autodrive/script/result.mp4', fourcc, fps, (1280, 720))
    while True:
        success, img = cap.read()
        img = cv2.cvtColor(cv2.resize(img, (1280, 720)), cv2.COLOR_BGR2RGB)
        #image = cv2.imread('test.jpg')
        img1, lanes, bev_lanes, bev_img, middle_lane = isnet.forward(img)
        cv2.imshow('demo', img1)
        out.write(img1)
        if cv2.waitKey(1) & 0xFF == ord('z'):
            cv2.waitKey(0)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
'''
