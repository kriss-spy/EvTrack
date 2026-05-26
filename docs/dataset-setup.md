# 数据集准备

## VisEvent

[VisEvent: Reliable object tracking via collaboration of frame and event flows](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)

IEEE Transactions on Cybernetics, 2023

## COESOT

[Revisiting color-event based tracking: A unified network, dataset, and metric](https://github.com/Event-AHU/COESOT)

Pattern Recognition, 2025

## 本地路径

数据集体积较大，不放入版本控制。建议下载后在本项目根目录创建符号链接或配置绝对路径：

```bash
# 示例：创建符号链接
ln -s /path/to/VisEvent data/VisEvent
ln -s /path/to/COESOT data/COESOT
```

或在各 tracker 的配置文件中直接指定数据集根目录。
