# Self-driving-Truck-in-Euro-Truck-Simulator2
# 简介
This is a self-driving truck test in Euro Truck Simulator2. I only test on windows11 + python3.8 + CUDA10.2.
### Perception
Use yolov6 to detect objects, clrnet to detect lane

### Control
Use PID and pure pursuit to control

### Plan
Use finite state machine

# 使用方法
1.开始前，请确保CUDA, cudnn, tensorrt安装完成。没安装的话，请先安装CUDA，CUDA的版本决定cudnn, tensorrt, 以及一些python库的版本。CUDA版本由显卡决定，不一定和我一样，安装尽量高的版本。

2.车道线检测与物体检测的pt文件和onnx文件已提供，PaddleOCR权重文件需前往PaddleOCR官方仓库下载。

3.下载[onnx权重文件](https://github.com/Yutong-gannis/Self-driving-Truck-in-Euro-Truck-Simulator2/releases/tag/v1.0)，tensorrt文件须用onnx文件进行转换。CLRNet权重转换方法可参考[CLRNet-onnxruntime-and-tensorrt-demo](https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo)，YOLOV6转换方法`python tools/export.py -o [onnx文件路径] -e [engine文件路径] --end2end`

例如`python tools/export.py -o yolov6n_bdd_40.onnx -e yolov6n_bdd_40.engine --end2end`

也可参考[TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)。

4.须自行下载vjoy虚拟手柄软件。

5.因为时间原因，未尝试更多的环境，python环境尽量与requirements.txt中保持一致。

6.main.py中设置的根目录设置为自己电脑上的文件根目录

终端输入`python script/main.py`以运行。

# 如何从零开始构建环境

### 所依赖的项目简介

- Python
  依赖的包支持的最小版本各有不同，推荐使用3.8版本。如使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来构建环境会节省很多时间。

- CUDA
  [官网地址](https://developer.nvidia.com/cuda-toolkit)， 推荐使用 [10.2](https://developer.nvidia.com/cuda-10.2-download-archive) 版本，次选[11.7.0](https://developer.nvidia.com/cuda-11-7-0-download-archive) 版本，其他版本支持情况未知。

  版本 `v12.0` 现阶段 `PyTorch` 等依赖的包还未支持。

- cuDNN (For CUDA 11.x, 10.2)
  [8.7 下载地址](https://developer.nvidia.com/rdp/cudnn-download)，跟CUDA版本匹配。其他介绍请参照 [cuDNN 安装手册](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)。

- TensorRT (For CUDA 11.x, 10.2)
  [8.x 下载地址](https://developer.nvidia.com/nvidia-tensorrt-8x-download)，跟CUDA版本匹配，推荐使用 `8.4.2.4(TensorRT 8.4 GA Update 1)` 版本。其他介绍请参照 [TensorRT 安装手册](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

- CUDA、cuDNN、TensorRT 请结合各自官方文档自行配置系统的 Path。

- 重新打开 PowerShell ，输入以下命令查看环境是否配置成功。

    ```bash
    PS C:\> python -V
    Python 3.8.10
    PS C:\> pip -V
    pip 22.3.1 from <python安装路径>\lib\site-packages\pip (python 3.8)
    PS C:\> nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
    Cuda compilation tools, release 10.2, V10.2.89
    PS C:\> trtexec.exe -h
    === Model Options ===
    <省略之后的输出>
    ```

- 稍微介绍一下几个依赖包的版本情况

    - PyTorch

        CUDA 是 10.2 的话，windows 下最高支持的版本为 `1.10.2+cu102`。

        ```bash
        python -m pip install torch==1.10.2+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
        ```

        CUDA 是 11.7 的话可以使用 `1.13.1+cu117`。

        ```bash
        python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
        ```

    - numpy

        从 1.24 版本开始删除了 `numpy.float` 等方法，因此需要使用 1.24 之前的版本。这里直接锁定使用 1.23.5。

        ```bash
        python -m pip install numpy==1.23.5
        ```

    - tensorrt

        依赖包请从tensorrt的所在文件夹中进行安装。

        ```bash
        # 注意这是在 windows 下的写法
        python -m pip install $env:TENSORRT_PATH\python\tensorrt-8.4.2.4-cp38-none-win_amd64.whl
        ```

        `TENSORRT_PATH` 为系统环境变量，`8.4.2.4` 为 `TensorRT` 的版本，`cp38` 与 安装的 `python` 的版本一致。

    - lap

        需要先安装 numpy 成功之后才能安装该依赖包。

        ```bash
        python -m pip install numpy==1.23.5 lap
        ```

### 安装依赖包

在项目根目录下运行以下命令。

```bash
# 注意 tensorrt 以及 python 的版本
python -m pip install $env:TENSORRT_PATH\python\tensorrt-8.4.2.4-cp38-none-win_amd64.whl
# cuda v10.2
python -m pip install -r requirements.txt
# cuda v11.7
# python -m pip install -r requirements_cu117.txt
python -m pip install -r last_requirements.txt
```

查看 PyTorch 是否为 cuda 版本。

```
PS C:\> python
Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
```

### 构建 TensorRT 文件

- 下载 [onnx 权重文件](https://github.com/Yutong-gannis/Self-driving-Truck-in-Euro-Truck-Simulator2/releases)，放在 `<PROJECT_PATH>/engines` 文件夹下（手动创建）。

- 安装权重转换所依赖的包

    ```bash
    python -m pip install -r ./tools/requirements.txt
    ```

- 构建 YOLOV6 的 TensorRT 文件

    ```bash
    python ./tools/export.py -o ./engines/yolov6s_bdd_60.onnx -e ./engines/yolov6s_bdd_60.engine --end2end
    ```

- 构建 CLRNet 的 TensorRT 文件

    ```bash
    polygraphy surgeon sanitize ./engines/llamas_dla34.onnx --fold-constants --output ./engines/llamas_dla34_tmp.onnx
    trtexec --onnx=./engines/llamas_dla34_tmp.onnx --saveEngine=./engines/llamas_dla34.engine
    ```

### 已测试的环境

Windows11 + Python3.8 的条件下在如下环境中运行成功。

- CUDA 10.2 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.7 + cuDNN 8.7.0 + TensorRT 8.4.2.4

PS: 修改完系统变量后需要重新打开一个新的 PowerShell。

# 如何自动驾驶

### 安装虚拟手柄

从[项目地址](https://sourceforge.net/projects/vjoystick/)下载 vJoy，双击安装。

### 设置游戏

- 选项 - 图像，取消全屏模式，设置合适的分辨率。
- 选项 - 控制，选择 `键盘 + vJoy Device`。
- 把游戏窗口移动到合适的位置。

# reference
[CLRNet](https://github.com/Turoad/CLRNet)

[CLRNet-onnxruntime-and-tensorrt-demo](https://github.com/xuanandsix/CLRNet-onnxruntime-and-tensorrt-demo)

[YOLOV6](https://github.com/meituan/YOLOv6)

[TensorRT-For-YOLO-Series](https://github.com/Linaom1214/TensorRT-For-YOLO-Series)

[pyvjoy](https://github.com/tidzo/pyvjoy)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
