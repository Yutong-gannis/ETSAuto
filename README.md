# Self-driving-Truck-in-Euro-Truck-Simulator2
This is a self-driving truck test in Euro Truck Simulator2. I only test on windows11 + python3.8 + CUDA10.2.

# Perception
Use yolov6 to detect objects, clrnet to detect lane

# Control
Use PID and pure pursuit to control

# Plan
Use finite state machine


# 使用方法
1.车道线检测与物体检测的pt文件和onnx文件已提供，PaddleOCR权重文件需前往PaddleOCR官方仓库下载。

2.tensorrt文件须用onnx文件进行转换

2.须自行下载vjoy虚拟手柄软件。

3.因为时间原因，未尝试更多的环境，python环境尽量与requirements.txt中保持一致。

3.main.py中设置的根目录设置为自己电脑上的文件根目录

终端输入`python script/main.py`以运行。

# Run
`python script/main.py`
