# 1 How to Build Environment from Scratch

### Introduction to Dependencies

- Python

  The minimum supported versions vary for different packages. It's recommended to use version 3.8 or higher. Using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to build the environment can save a lot of time.

- CUDA

  [Official Website](https://developer.nvidia.com/cuda-toolkit) - Recommended versions are [10.2](https://developer.nvidia.com/cuda-10.2-download-archive), and as a secondary option, [11.7.0](https://developer.nvidia.com/cuda-11-7-0-download-archive). Compatibility with other versions is unknown. Version `v12.0` is not yet supported by packages like `PyTorch`.

- cuDNN (For CUDA 11.x, 10.2)

  [Version 8.7 Download](https://developer.nvidia.com/rdp/cudnn-download) - Match the version with CUDA. For installation instructions, refer to the [cuDNN Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html).

- TensorRT (For CUDA 11.x, 10.2) (Optional)

  [Version 8.x Download](https://developer.nvidia.com/nvidia-tensorrt-8x-download) - Match the version with CUDA. Recommend using `8.4.2.4 (TensorRT 8.4 GA Update 1)`. For installation instructions, refer to the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html).

- For CUDA, cuDNN, and TensorRT (Optional), configure the system's Path based on their respective official documentation.

- Reopen PowerShell and enter the following commands to verify successful configuration:

    ```bash
    PS C:\> python -V
    Python 3.8.10
    PS C:\> pip -V
    pip 22.3.1 from <python-install-path>\lib\site-packages\pip (python 3.8)
    PS C:\> nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Wed_Oct_23_19:32:27_Pacific_Daylight_Time_2019
    Cuda compilation tools, release 10.2, V10.2.89
    PS C:\> trtexec.exe -h
    === Model Options ===
    <omitting the rest of the output>
    ```

- Briefly introduce the versions of some dependencies:

    - PyTorch

        For CUDA 10.2, the highest supported version on Windows is `1.10.2+cu102`.

        ```bash
        python -m pip install torch==1.10.2+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
        ```

        For CUDA 11.7, you can use `1.13.1+cu117`.

        ```bash
        python -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
        ```

    - numpy
        ```bash
        python -m pip install numpy
        ```

    - tensorrt

        Install the dependency package from the tensorrt directory.

        ```bash
        # Note: This is the syntax for Windows PowerShell
        python -m pip install $env:TENSORRT_PATH\python\tensorrt-8.4.2.4-cp38-none-win_amd64.whl
        ```

        `TENSORRT_PATH` is the system environment variable, `8.4.2.4` is the version of `TensorRT`, and `cp38` matches the installed `python` version.

    - lap

        You need to install `numpy` successfully before installing this package.

        ```bash
        python -m pip install numpy<1.24 lap
        ```

### Download Software
Download the latest version of the software to your local computer from the [release page](https://github.com/Yutong-gannis/ETSAuto/releases).

### Install Dependency Packages

Run the following commands in the project root directory.

```bash
# Make sure to consider TensorRT and Python versions
python -m pip install $env:TENSORRT_PATH\python\tensorrt-8.4.2.4-cp38-none-win_amd64.whl

# For CUDA v10.2
python -m pip install -r requirements.txt
# For CUDA v11.7 (Ignore if using CUDA v10.2)
python -m pip install -r requirements_cu117.txt

python -m pip uninstall -y opencv-python-headless
python -m pip install -r last_requirements.txt
```

Check if PyTorch is the CUDA version:

```
PS C:\> python
Python 3.8.10 (tags/v3.8.10:3d8993a, May  3 2021, 11:48:03) [MSC v.1928 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.cuda.is_available()
True
```

### Download Weight Files
Download the [ONNX weight file](https://github.com/Yutong-gannis/ETSAuto/releases/download/v2.0/ep049.onnx) and place it in the `<PROJECT_PATH>/weights` folder (create manually).

### Tested Environments

Successfully ran on Windows 11 + Python 3.8 under the following conditions:

- CUDA 10.2 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.2 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.7 + cuDNN 8.7.0 + TensorRT 8.4.2.4
- CUDA 11.7 + cuDNN 8.7.0 + TensorRT 8.4.3.1
- CUDA 10.2 + cuDNN 8.7.0

# 2 Game Settings
### Install Virtual Joystick

Download vJoy from the [project link](https://sourceforge.net/projects/vjoystick/) and install by double-clicking.

### Install Mods

Download mods from the Steam Workshop:

- Google Maps Navigation Night Version
- SISL's Route Adviser

### In-Game Settings

- Options - Graphics: Disable full-screen mode, set the resolution to 1360x768.
- Options - Controls: Choose `Keyboard + vJoy Device`.
- Options - Keys and Buttons: Unassign the numeric keypad **num 0~9**, **Up**, **Down**, **Left**, **Right** keys; these keys are used in the program.
- Move the game window to the upper-left corner of the screen for screen capture (note: fine-tune left and right by a few pixels to ensure the vehicle is centered
