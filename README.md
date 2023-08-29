# ETSAuto2
### 目录
+ [简介](#简介)
+ [功能](#功能)
+ [Todo](#Todo)

## 简介
ETSAuto 2是可以运行在欧洲卡车模拟2上的卡车辅助驾驶系统，ETSAuto 2已经实现了车道保持辅助（LKA）和低速状态下的变道辅助（LCA）。ETSAuto 2采用纯python语言，运行于windows系统，感知模块采用onnruntime推理（放弃了之前的TensorRT推理框架），目前支持在Nvidia显卡上进行加速，未来可以支持使用AMD显卡进行加速。

## 功能

### 运行
双击`ETSAuto.bat`

# Todo
- [x] LCA
- [ ] LCW
- [ ] SAS
- [ ] TLR
