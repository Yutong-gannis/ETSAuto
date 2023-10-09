# 1 如何从零开始构建环境

### 1） 环境配置

- Python

  依赖的包支持的最小版本各有不同，推荐使用3.8以上版本。如使用 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 来构建环境会节省很多时间。

- CUDA

  [官网地址](https://developer.nvidia.com/cuda-toolkit)， 推荐使用 [10.2](https://developer.nvidia.com/cuda-10.2-download-archive) 版本，次选[11.7.0](https://developer.nvidia.com/cuda-11-7-0-download-archive) 版本，其他版本支持情况未知。

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

### 2）项目安装
+ 下载项目
  
  从[发布页面](https://github.com/Yutong-gannis/ETSAuto/releases)下载项目最新版本到本地电脑。

+ 安装依赖包
  
  在项目根目录下运行以下命令。
  ```bash
  python -m pip install -r requirements.txt
  ```

+ 下载权重文件
  
  从[release](https://github.com/Yutong-gannis/ETSAuto/releases)下载最新的bevlanedet和yolov8n模型，放在 `<PROJECT_PATH>/weights` 文件夹下（手动创建）。

### 3）已测试的环境

Windows11 + Python3.8 的条件下在如下环境中运行成功。

- CUDA 10.2 + cuDNN 8.7.0
- CUDA 11.2 + cuDNN 8.7.0
- CUDA 11.7 + cuDNN 8.7.0

# 2 游戏设置
### 1)安装虚拟手柄

从[项目地址](https://sourceforge.net/projects/vjoystick/)下载 vJoy，双击安装。安装过程中，会提示重启，请按要求重启并完成剩余安装操作。

### 2)安装插件
在[thirdparty/ets2-sdk-plugin](https://github.com/Yutong-gannis/ETSAuto/tree/v2.x/thirdparty/ets2-sdk-plugin)目录下，根据设备选择WIN32或WIN64文件夹，将文件夹下的**ets2-telemetry.dll**复制到欧卡安装根目录下的bin/win_x86/plugins（若没有plugins文件夹，则手动创建）。开启游戏后，若游戏出现警告，则证明安装完成。

### 3)安装mod

通过创意工坊下载mod

- Google Maps Navigation Night Version
- SISL's Route Adviser

### 4)游戏内设置

- 选项 - 图像，取消全屏模式，分辨率设置为 1360x768。
- 选项 - 控制，选择 `键盘 + vJoy Device`。
- 选项 - 按键和按钮，将其中小键盘下的**num 0~9**、**Up**、**Down**、**Left**、**Right**设置为取消分配，程序中需要用到这些键。
- 把游戏窗口移动到屏幕左上角合适的位置，方便程序对屏幕进行获取（注意：左右精调几个像素才能保证车辆行驶在道路正中间）。
- 按F5把导航地图调成最大。

PS: 修改完系统变量后需要重新打开一个新的 PowerShell。
