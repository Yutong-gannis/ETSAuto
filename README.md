# ETSAuto2 🚚
### 目录
+ [简介](#简介)
+ [功能](#功能)
+ [环境搭建](#环境搭建)
+ [使用说明](#使用说明)
+ [计划](#计划)
+ [参考](#参考)

## 简介
ETSAuto 2是可以运行在欧洲卡车模拟2上的卡车辅助驾驶系统，ETSAuto 2已经实现了车道保持辅助（LKA）和低速状态下的变道辅助（LCA）。ETSAuto 2采用纯python语言，运行于windows系统，感知模块采用onnruntime推理（支持TensorRT推理框架），目前支持在Nvidia显卡上进行加速，未来可以支持使用AMD显卡进行加速。

## 功能
| 功能               | 支持  | 说明 |
| ---                | :---: | --- |
| 车道保持辅助（LKA） | ✓     | v < 75km/h |
| 变道辅助（LCA）     | ✓    | v < 50km/h |
| 前向碰撞预警（LCW） | ✗    |     |
| 自适应巡航（ACC）   | ✗    |     |

## 环境搭建
环境搭建请参考[BUILD.md](https://github.com/Yutong-gannis/ETSAuto/blob/v2.0dev/BUILD.md)

考虑到应用对显卡的兼容性问题，在v2.0以后的版本中将主要使用onnxruntime进行推理，目前可以支持nvidia显卡，未来希望可以支持amd显卡。但目前手里没有amd显卡设备，开发者们可以尝试在amd显卡上进行构建。项目中仍留有tensorrt推理的接口，以保证必要的感知响应频率。由于屏幕捕获程序和vjoy控制程序的原因，目前程序仍仅支持windows系统，欢迎各位开发者提供上述两个程序在linux系统上的替代方案。

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

![a6x18041cro5ffnvur8sb1c](https://github.com/Yutong-gannis/ETSAuto/assets/69740611/11d36472-3cfa-42bc-b8ef-f71576f872c7)

## 计划
- [x] LCA
- [x] LKA
- [ ] LCW
- [ ] SAS
- [ ] TLR

## 参考
[pyvjoy](https://github.com/tidzo/pyvjoy)

[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

[Bev-Lanedet](https://github.com/gigo-team/bev_lane_det)
