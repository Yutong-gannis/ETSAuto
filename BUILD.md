# 1 如何从零开始构建环境

### 所依赖的项目简介

- Python

  依赖的包支持的最小版本各有不同，推荐使用3.8以上版本。如使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来构建环境会节省很多时间。

- CUDA

  [官网地址](https://developer.nvidia.com/cuda-toolkit)， 推荐使用 [10.2](https://developer.nvidia.com/cuda-10.2-download-archive) 版本，次选[11.7.0](https://developer.nvidia.com/cuda-11-7-0-download-archive) 版本，其他版本支持情况未知。版本 `v12.0` 现阶段 `PyTorch` 等依赖的包还未支持。

- cuDNN (For CUDA 11.x, 10.2)

  [8.7 下载地址](https://developer.nvidia.com/rdp/cudnn-download)，跟CUDA版本匹配。其他介绍请参照 [cuDNN 安装手册](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)。

- TensorRT (For CUDA 11.x, 10.2)（可选）

  [8.x 下载地址](https://developer.nvidia.com/nvidia-tensorrt-8x-download)，跟CUDA版本匹配，推荐使用 `8.4.2.4(TensorRT 8.4 GA Update 1)` 版本。其他介绍请参照 [TensorRT 安装手册](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)。

- CUDA、cuDNN、TensorRT（可选） 请结合各自官方文档自行配置系统的 Path。

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
        ```bash
        python -m pip install numpy
        ```

    - tensorrt

        依赖包请从tensorrt的所在文件夹中进行安装。

        ```bash
        # 注意这是在 windows 的 PowerShell 下的写法
        python -m pip install $env:TENSORRT_PATH\python\tensorrt-8.4.2.4-cp38-none-win_amd64.whl
        ```

        `TENSORRT_PATH` 为系统环境变量，`8.4.2.4` 为 `TensorRT` 的版本，`cp38` 与 安装的 `python` 的版本一致。

    - lap

        需要先安装 numpy 成功之后才能安装该依赖包。

        ```bash
        python -m pip install numpy<1.24 lap
        ```

### 安装依赖包

在项目根目录下运行以下命令。

```bash
# 注意 tensorrt 以及 python 的版本
python -m pip install $env:TENSORRT_PATH\python\tensorrt-8.4.2.4-cp38-none-win_amd64.whl

# cuda v10.2
python -m pip install -r requirements.txt
# cuda v11.7(cuda为v10.2的忽略此步)
python -m pip install -r requirements_cu117.txt

python -m pip uninstall -y opencv-python-headless
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

### 下载权重文件
下载 [onnx 权重文件](https://github.com/Yutong-gannis/ETSAuto/releases/tag/untagged-6d929c6606d0e15085e8)，放在 `<PROJECT_PATH>/weights` 文件夹下（手动创建）。

### 已测试的环境

Windows11 + Python3.8 的条件下在如下环境中运行成功。

- CUDA 10.2 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.2 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.7 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.7 + cuDNN 8.7.0 + TensorRT 8.4.3.1
- CUDA 10.2 + cuDNN 8.7.0

# 2 游戏设置
### 安装虚拟手柄

从[项目地址](https://sourceforge.net/projects/vjoystick/)下载 vJoy，双击安装。

### 安装mod

通过创意工坊下载mod

- Google Maps Navigation Night Version
- SISL's Route Adviser

### 游戏内设置

- 选项 - 图像，取消全屏模式，分辨率设置为 1360x768。
- 选项 - 控制，选择 `键盘 + vJoy Device`。
- 选项 - 按键和按钮，将其中小键盘下的**num 0~9**、**Up**、**Down**、**Left**、**Right**设置为取消分配，程序中需要用到这些键。
- 把游戏窗口移动到屏幕左上角合适的位置，方便程序对屏幕进行获取（注意：左右精调几个像素才能保证车辆行驶在道路正中间）。
- 按F5把导航地图调成最大。

PS: 修改完系统变量后需要重新打开一个新的 PowerShell。
