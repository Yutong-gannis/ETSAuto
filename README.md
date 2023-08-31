<div align="center">
  <img src="https://github.com/Yutong-gannis/ETSAuto/assets/69740611/9ebe1832-46bc-408a-a2a5-692985454c27" width="400" height="400"/>

  ![GitHub](https://img.shields.io/github/license/Yutong-gannis/ETSAuto)
  ![build](https://img.shields.io/badge/build-passing-green)
  ![python3](https://img.shields.io/badge/python3-pass-green)
  ![pr](https://img.shields.io/badge/PRs-welcome-brightgreen)
  ![GitHub (Pre-)Release Date](https://img.shields.io/github/release-date-pre/Yutong-gannis/ETSAuto)
  ![GitHub release (with filter)](https://img.shields.io/github/v/release/Yutong-gannis/ETSAuto)
  ![GitHub tag (with filter)](https://img.shields.io/github/v/tag/Yutong-gannis/ETSAuto)
  ![docs](https://img.shields.io/badge/docs-latest-blue)
  ![GitHub Repo stars](https://img.shields.io/github/stars/Yutong-gannis/ETSAuto)
  ![languages](https://img.shields.io/github/languages/top/yutong-gannis/ETSAuto)
  ![GitHub forks](https://img.shields.io/github/forks/Yutong-gannis/ETSAuto)
  
  ![GitHub issues](https://img.shields.io/github/issues/Yutong-gannis/ETSAuto)
  ![GitHub closed issues](https://img.shields.io/github/issues-closed/Yutong-gannis/ETSAuto)
  ![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Yutong-gannis/ETSAuto)

</div>

<div align="center">
    
简体中文 | [English](https://github.com/Yutong-gannis/ETSAuto/blob/v2.x/README_en.md)

</div>

### 目录
- [ETSAuto2 🚚](#etsauto2-)
    - [目录](#目录)
  - [简介](#简介)
  - [功能](#功能)
  - [环境搭建](#环境搭建)
  - [使用说明](#使用说明)
  - [赞助](#赞助)
  - [计划](#计划)
  - [参考](#参考)

## 简介
ETSAuto 2是可以运行在欧洲卡车模拟2上的卡车辅助驾驶系统，ETSAuto 2已经实现了车道保持辅助（LKA）和低速状态下的变道辅助（LCA）。ETSAuto 2采用纯python语言，运行于windows系统，感知模块采用onnruntime推理，目前支持在Nvidia显卡上进行加速，未来可以支持使用AMD显卡进行加速。

## 功能
| 功能               | 支持  | 说明 |
| ---                | :---: | --- |
| 车道保持辅助（LKA） | ✓     | v < 75km/h |
| 变道辅助（LCA）     | ✓    | v < 50km/h |
| 前向碰撞预警（LCW） | ✗    |     |
| 自适应巡航（ACC）   | ✗    |     |

## 环境搭建
环境搭建请参考[BUILD.md](https://github.com/Yutong-gannis/ETSAuto/blob/v2.0dev/BUILD.md)

考虑到应用对显卡的兼容性问题，在v2.0以后的版本中将主要使用onnxruntime进行推理，目前可以支持nvidia显卡，未来希望可以支持amd显卡。但目前手里没有amd显卡设备，开发者们可以尝试在amd显卡上进行构建。由于屏幕捕获程序和vjoy控制程序的原因，目前程序仍仅支持windows系统，欢迎各位开发者提供上述两个程序在linux系统上的替代方案。

## 使用说明
+ 程序入口
  
  双击`ETSAuto.bat`打开程序

+ 按键说明

  为了方便操作，功能采用键盘进行控制

  | 按键   | 功能     | 支持  | 说明 |
  | :---:  | :---:    | :---: | ---    |
  | &darr; | 手动     | ✓     |     |
  | &larr; | 横向控制 | ✓     |     |
  | &rarr; | 纵向控制 | ✓     |     |
  | &uarr; | 辅助驾驶 | ✓     |     |
  | num 0  | 直行     | ✓     | v < 75km/h |
  | num 1  | 左转     | ✗     |     |
  | num 3  | 右转     | ✗     |     |
  | num 4  | 左变道   | ✓     | v < 50km/h |
  | num 6  | 右变道   | ✓     | v < 50km/h |
  | ctrl+q | 退出     | ✓     |     |



## 赞助
如果您喜欢这个项目，并希望我继续下去，可以考虑赞助我！感谢所有的爱和支持！

+ 支付宝

![a6x18041cro5ffnvur8sb1c](https://github.com/Yutong-gannis/ETSAuto/assets/69740611/11d36472-3cfa-42bc-b8ef-f71576f872c7)

+ 微信

![94214ceaca44085494ab079ce946ad4](https://github.com/Yutong-gannis/ETSAuto/assets/69740611/f455eab6-76f4-4a56-b5a8-5ec313f506f1)


## 计划
- [x] LCA
- [x] LKA
- [ ] LCW
- [ ] SAS
- [ ] TLR

## 参考
[pyvjoy](https://github.com/tidzo/pyvjoy)

[Bev-Lanedet](https://github.com/gigo-team/bev_lane_det)
