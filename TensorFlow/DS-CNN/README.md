# DSCNN
## 概述
迁移[DSCNN](https://github.com/ARM-software/ML-KWS-for-MCU/) 到ascend910平台,

得到的结果和论文的对比，
使用的是albert_v2版本的预训练模型
|  | Acc| 
| :-----| ----: | 
| Ascend | **95.32**|
| 论文| 95.4 |

## Requirements
- Tensorflow 1.15
- Huawei Ascend 910

## 代码路径解释

```shell
ds-cnn
└─ 
  ├─summary 存放TensorBoard event日志文件
  ├─speech_command_train 存放checkpoint文件 
  ├─train_testcase.sh 启动脚本
  ├─fold_batchnorm.py
  ├─freeze.py
  ├─input_data.py
  ├─label_wav.py
  ├─models.py
  ├─quant_models.py
  ├─quant_test.py
  ├─silence.wav
  ├─test.py
  ├─test_pb.py
  ├─train.py
  |─train_commands.txt 不同模型训练命令文档
```

---

## 准备数据和模型
程序代码会自动下载数据集

## 参数解释
	
--batch_size = batchsize   
--data_dir = 数据集目录   
--how_many_training_steps = 训练step数    
--learning_rate = 学习率   
--summaries_dir = 训练产生的event日志存放目录   
--train_dir = 训练结果存放目录    
--model_size_info = 模型大小信息
--window_size_ms = 窗口尺寸
--window_stride_ms = 窗口步长
--dct_coefficient_count = 系数计数

## 训练

./train_testcase.sh
