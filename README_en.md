<div align="center">
  <img src="https://github.com/Yutong-gannis/ETSAuto/assets/69740611/9ebe1832-46bc-408a-a2a5-692985454c27" width="400" height="400"/>

  ![GitHub](https://img.shields.io/github/license/Yutong-gannis/ETSAuto)
  ![build](https://img.shields.io/badge/build-passing-green)
  ![python3](https://img.shields.io/badge/python3-pass-green)
  ![pr](https://img.shields.io/badge/PRs-welcome-brightgreen)
  ![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Yutong-gannis/ETSAuto)
  ![docs](https://img.shields.io/badge/docs-latest-blue)
  ![GitHub Repo stars](https://img.shields.io/github/stars/Yutong-gannis/ETSAuto)
  ![languages](https://img.shields.io/github/languages/top/yutong-gannis/ETSAuto)
  ![GitHub forks](https://img.shields.io/github/forks/Yutong-gannis/ETSAuto)
  
  ![GitHub release (with filter)](https://img.shields.io/github/v/release/Yutong-gannis/ETSAuto)
  ![GitHub tag (with filter)](https://img.shields.io/github/v/tag/Yutong-gannis/ETSAuto)
  ![GitHub issues](https://img.shields.io/github/issues/Yutong-gannis/ETSAuto)
  ![GitHub closed issues](https://img.shields.io/github/issues-closed/Yutong-gannis/ETSAuto)
  ![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Yutong-gannis/ETSAuto)

</div>

<div align="center">

[简体中文](https://github.com/Yutong-gannis/ETSAuto/blob/v2.x/README.md) | English

</div>

### Table of Contents
+ [Introduction](#introduction)
+ [Features](#features)
+ [Environment Setup](#environment-setup)
+ [Usage Instructions](#usage-instructions)
+ [Plans](#plans)
+ [References](#references)

## Introduction
ETSAuto 2 is a truck-assistance driving system that runs on Euro Truck Simulator 2. ETSAuto 2 has implemented Lane Keeping Assistance (LKA) and Lane Change Assistance (LCA) at low speeds. ETSAuto 2 is developed in pure Python and runs on Windows. The perception module uses ONNX Runtime for inference. Currently, it supports acceleration on Nvidia graphics cards, and future support for AMD graphics cards is planned.

## Features
| Feature               | Support | Description |
| ---                   | :---:   | ---         |
| Lane Keeping (LKA)    | ✓       | v < 75km/h  |
| Lane Change (LCA)     | ✓       | v < 50km/h  |
| Forward Collision Warning (LCW) | ✗ |             |
| Adaptive Cruise Control (ACC)   | ✗ |             |

## Environment Setup
For environment setup, refer to [BUILD_en.md](https://github.com/Yutong-gannis/ETSAuto/blob/v2.x/BUILD_en.md)

Considering compatibility with graphics cards, from version 2.0 onwards, ONNX Runtime will be mainly used for inference. Nvidia graphics cards are currently supported, and support for AMD graphics cards is planned. However, since there is no AMD graphics card available at the moment, developers are encouraged to attempt building on AMD graphics cards. Due to reasons related to screen capture and vjoy control programs, the program currently supports only Windows. Developers are welcome to provide alternative solutions for these two codes on Linux systems.

## Usage Instructions
+ Program Entry

  Double click `ETSAuto.bat` to open the program.

+ Key Instructions

  To facilitate operation, keyboard controls are used for functionalities.

  | Key      | Function  | Support | Description |
  | :---:    | :---:     | :---:   | ---         |
  | &darr;   | Manual    | ✓       |             |
  | &larr;   | Lateral   | ✓       |             |
  | &rarr;   | Longitudinal | ✓     |             |
  | &uarr;   | Assistance | ✓       |             |
  | num 0    | Straight  | ✓       | v < 75km/h  |
  | num 1    | Left Turn | ✗       |             |
  | num 3    | Right Turn| ✗       |             |
  | num 4    | Left Lane Change | ✓ | v < 50km/h  |
  | num 6    | Right Lane Change | ✓ | v < 50km/h |
  | ctrl+q   | Exit      | ✓       |             |

## Sponsorship
If you like this project and want me to continue, consider sponsoring me! Thanks for all the love and support!

+ Alipay

![a6x18041cro5ffnvur8sb1c](https://github.com/Yutong-gannis/ETSAuto/assets/69740611/11d36472-3cfa-42bc-b8ef-f71576f872c7)

+ Wechat

![94214ceaca44085494ab079ce946ad4](https://github.com/Yutong-gannis/ETSAuto/assets/69740611/f455eab6-76f4-4a56-b5a8-5ec313f506f1)


## Plans
- [x] LCA
- [x] LKA
- [ ] LCW
- [ ] SAS
- [ ] TLR

## References
[pyvjoy](https://github.com/tidzo/pyvjoy)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[Bev-Lanedet](https://github.com/gigo-team/bev_lane_det)
