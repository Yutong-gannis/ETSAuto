# Self-driving-Truck-in-Euro-Truck-Simulator2
This is a self-driving truck test in Euro Truck Simulator2. I only test on windows11 + python3.8 + CUDA10.2.

# Perception
Use yolov6 to detect objects, clrnet to detect lane

# Control
Use PID and pure pursuit to control

# Plan
Use finite state machine


# Run
`python script/main.py`

# 使用方法
1.开始前，请确保CUDA, cudnn, tensorrt安装完成。没安装的话，请先安装CUDA，CUDA的版本决定cudnn, tensorrt, 以及一些python库的版本。CUDA版本由显卡决定，不一定和我一样，安装尽量高的版本。

2.车道线检测与物体检测的pt文件和onnx文件已提供，PaddleOCR权重文件需前往PaddleOCR官方仓库下载。

3.下载[onnx权重文件](https://github.com/Yutong-gannis/Self-driving-Truck-in-Euro-Truck-Simulator2/releases/tag/v1.0)，tensorrt文件须用onnx文件进行转换。CLRNet权重转换方法可参考[CLRNet-onnxruntime-and-tensorrt-demo](https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo)，YOLOV6转换方法`python tools/export.py -o [onnx文件路径] -e [engine文件路径] --end2end`

例如`python tools/export.py -o yolov6n_bdd_40.onnx -e yolov6n_bdd_40.engine --end2end`

也可参考[https://github.com/Linaom1214/TensorRT-For-YOLO-Series]。

4.须自行下载vjoy虚拟手柄软件。

5.因为时间原因，未尝试更多的环境，python环境尽量与requirements.txt中保持一致。

6.main.py中设置的根目录设置为自己电脑上的文件根目录

终端输入`python script/main.py`以运行。

# reference
[CLRNet](https://github.com/Turoad/CLRNet)

[CLRNet-onnxruntime-and-tensorrt-demo](https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo)

[YOLOV6](https://github.com/meituan/YOLOv6)

[TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)

[pyvjoy](https://github.com/tidzo/pyvjoy)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
