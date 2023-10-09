# 1 Building the Environment

### 1) Environment Configuration

- Python

  Different packages have varying minimum version requirements, but it is recommended to use Python 3.8 or higher. Using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to build the environment can save a lot of time.

- CUDA

  [Official website](https://developer.nvidia.com/cuda-toolkit). It is recommended to use version [10.2](https://developer.nvidia.com/cuda-10.2-download-archive) as the first choice and [11.7.0](https://developer.nvidia.com/cuda-11-7-0-download-archive) as the second choice. Support for other versions is unknown.

- cuDNN (For CUDA 11.x, 10.2)

  [Download link for 8.7](https://developer.nvidia.com/rdp/cudnn-download), matching the CUDA version. For more details, refer to the [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

- TensorRT (For CUDA 11.x, 10.2) (Optional)

  [Download link for 8.x](https://developer.nvidia.com/nvidia-tensorrt-8x-download), matching the CUDA version. It is recommended to use version `8.4.2.4 (TensorRT 8.4 GA Update 1)`. For more details, refer to the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

- For CUDA, cuDNN, and TensorRT (optional), configure the system's Path as per their respective official documentation.

- Reopen PowerShell and enter the following commands to check if the environment is configured successfully.

    ```bash
    PS C:\> python -V
    Python 3.8.10
    PS C:\> pip -V
    pip 22.3.1 from <python install path>\lib\site-packages\pip (python 3.8)
    PS C:\> nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
    Cuda compilation tools, release 10.2, V10.2.89
    PS C:\> trtexec.exe -h
    === Model Options ===
    <output omitted>
    ```

### 2) Project Installation

+ Download the Project

  Download the latest version of the project to your local computer from the [release page](https://github.com/Yutong-gannis/ETSAuto/releases).

+ Install Dependencies

  Run the following command in the project's root directory.

  ```bash
  python -m pip install -r requirements.txt
  ```

+ Download Weight Files

  Download the latest bevlanedet and yolov8n models from [release](https://github.com/Yutong-gannis/ETSAuto/releases) and place them in the `<PROJECT_PATH>/weights` folder (create it manually if necessary).

### 3) Tested Environments

The project has successfully run on Windows 11 with Python 3.8 under the following environments:

- CUDA 10.2 + cuDNN 8.7.0
- CUDA 11.2 + cuDNN 8.7.0
- CUDA 11.7 + cuDNN 8.7.0

# 2 Game Setup

### 1) Install Virtual Joystick

Download vJoy from the [project's page](https://sourceforge.net/projects/vjoystick/), and install it by double-clicking. During the installation, you will be prompted to restart your computer. Please follow the instructions to restart and complete the remaining installation steps.

### 2) Install the Plugin

In the [thirdparty/ets2-sdk-plugin](https://github.com/Yutong-gannis/ETSAuto/tree/v2.x/thirdparty/ets2-sdk-plugin) directory, choose the WIN32 or WIN64 folder based on your device, then copy the **ets2-telemetry.dll** file to the bin/win_x86/plugins directory under the Euro Truck Simulator 2 installation root directory (create the plugins folder manually if it doesn't exist). After starting the game, if you receive a warning, it means the installation is complete.

### 3) Install Mods

Download mods via the Steam Workshop:

- Google Maps Navigation Night Version
- SISL's Route Adviser

### 4) In-Game Settings

- Options - Graphics, disable fullscreen mode, set the resolution to 1360x768.
- Options - Controls, select `Keyboard + vJoy Device`.
- Options - Keys and Buttons, unassign the keys on the numpad (num 0~9, Up, Down, Left, Right) as these keys are used in the program.
- Move the game window to the top-left corner of the screen for better screen capture (note: fine-tune left-right by a few pixels to ensure the vehicle is centered on the road).
- Press F5 to maximize the navigation map.

PS: After modifying system variables, you need to reopen a new PowerShell window.
