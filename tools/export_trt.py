import os
import time
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

start_time = time.time()
logging.info("Start export yolo weight")
os.system('python tools/pt2trt/yolo2trt.py -o weights/yolov6s_bdd_60.onnx -e weights/yolov6s_bdd_60.engine --end2end')
logging.info("Finish export yolo weight")

logging.info("Start export clrnet weight")
os.system('polygraphy surgeon sanitize weights/llamas_dla34.onnx --fold-constants --output weights/llamas_dla34_t.onnx')
os.system('python tools/pt2trt/onnx2trt.py -o weights/llamas_dla34_t.onnx -e weights/llamas_dla34.engine')
logging.info("Finish export clrnet weight")

logging.info("Start export scene weight")
os.system('python tools/pt2trt/onnx2trt.py -o weights/weather_detector.onnx -e weights/weather_detector.engine')
logging.info("Finish export scene weight")
end_time = time.time()
logging.info("Export time: {}s".format(str(end_time-start_time)))
