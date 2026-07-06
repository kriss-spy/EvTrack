# EvTrack：基于事件相机的目标跟踪

> [English](README.md) | 简体中文

## 概述

EvTrack 是一个聚焦**事件相机单目标跟踪**的研究项目。事件相机具有微秒级时间分辨率和高动态范围，在高速、极端光照等传统帧相机失效的场景下具备天然优势。

本项目在 [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) 基准上复现并评估了两个 RGB-Event 多模态跟踪器：

- **[ViPT](https://github.com/jiawen-zhu/ViPT)**（CVPR 2023）——视觉提示多模态跟踪，并在此基础上做了在线模板改进。
- **[SDSTrack](https://github.com/hoqolo/SDSTrack)**（CVPR 2024）——自蒸馏对称适配器跟踪。

## ViPT

ViPT 的复现与在线模板改进代码位于独立 fork：

> **https://github.com/kriss-spy/ViPT** —— 分支
> [`vipt-improvement`](https://github.com/kriss-spy/ViPT/tree/vipt-improvement)

相对上游的关键改动与使用说明见 [`code/ViPT/README.md`](code/ViPT/README.md)。

### 在线模板改进

我们在原始 ViPT 跟踪器中加入了一对在线模板（RGB + event），并采用基于 SSIM 的门控更新规则——**无需额外训练**，直接使用原始权重。这是一种"补救性"措施：在部分 VisEvent 序列上能改善跟踪，但无法从根本上解决原始结果较差的情况，在某些场景下反而会降低性能。

演示 GIF（原始 vs. 改进）：

![visevent video0079](./gif/video0079.gif)
![compare 2](./gif/compare2.gif)
![compare 3](./gif/compare3.gif)
![compare 4](./gif/compare4.gif)

失败案例——在线模板有时会损害性能：

![failure case](./gif/fail1.gif)

### 进一步研究

- [ViPT 量化探索方案](docs/vipt-quantization-research.md) ——
  FP16/INT8 PTQ/QAT 路线图（Issue #9）。

## SDSTrack

在 VisEvent 测试集上复现 SDSTrack（319/320 序列；与官方 MATLAB 工具等价的评测协议，排除 absent 帧）：

| 跟踪器 | Success AUC | Precision @ 20px | 与论文差距 |
|---------|------------:|-----------------:|-----------|
| **SDSTrack**（复现） | **0.5829** | **0.7506** | 2% 以内 |
| SDSTrack（论文） | 0.597 | 0.767 | — |

- 代码与快速开始：[`code/SDSTrack/`](code/SDSTrack/)
- 结果、指标与复现日志：[`experiments/sdstrack/`](experiments/sdstrack/)
- 逐序列预测结果存档：Hugging Face 上的
  [`krisspy39/visevent-sdstrack-results`](https://huggingface.co/datasets/krisspy39/visevent-sdstrack-results)。
- VisEvent 复现完整报告：[`docs/事件相机目标跟踪.pdf`](docs/事件相机目标跟踪.pdf)

## 文档

项目全部文档的分类索引见 [`docs/README.md`](docs/README.md)。重点：

- [课程设计报告](docs/report.md) —— 待发布（Issue #27）
- [开题报告](docs/project-proposal.md)
- [数据集准备指南](docs/dataset-setup.md)
- [课程设计指南与评分标准](docs/course-project-guide.md)
- [事件相机综述阅读笔记](docs/Event-Based%20Vision.md)
- [SDSTrack 复现实验](experiments/sdstrack/README.md)

## 数据集

- [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark) ——
  主要基准；同时在
  [Hugging Face](https://huggingface.co/datasets/krisspy39/visevent) 上有镜像。
- [COESOT](https://github.com/Event-AHU/COESOT) —— 辅助数据集（待获取）。

下载细节见 [docs/dataset-setup.md](docs/dataset-setup.md)。

## 项目结构

```
.
├── code/               # 跟踪器实现
│   ├── SDSTrack/       #   SDSTrack 复现（上游 submodule + 评测脚本）
│   └── ViPT/           #   指向 kriss-spy/ViPT fork（vipt-improvement 分支）
├── experiments/        # 实验配置、指标、复现日志
├── scripts/            # 共享评测指标脚本
├── data/               # 数据集路径与说明（不含原始数据）
├── results/            # 评测输出、图表、视频
├── docs/               # 文档与指南
├── gif/                # 跟踪演示 GIF
├── requirements.txt
└── README.md
```

## 快速开始

### 通用准备

```bash
# 带子模块克隆
git clone --recurse-submodules https://github.com/kriss-spy/EvTrack

# 若已克隆：
git submodule update --init

# 安装基础依赖
pip install -r requirements.txt
```

### ViPT

```bash
# 单独克隆 fork
git clone -b vipt-improvement https://github.com/kriss-spy/ViPT

# 在 VisEvent 上评测
cd ViPT
bash eval_rgbe.sh --seq_home /path/to/VisEvent/test --save_dir ./RGBE_workspace/results
```

详见 [`code/ViPT/README.md`](code/ViPT/README.md)。

### SDSTrack

```bash
# 运行评测
python code/SDSTrack/sdstrack_eval.py --workspace /workspace/sdstrack

# 计算指标（与 MATLAB 等价的协议）
python scripts/eval_visevent_matlab.py --results <path> --dataset <path>
```

详见 [`code/SDSTrack/README.md`](code/SDSTrack/README.md)。

## 参考文献

[1] Zhu J, Lai S, Chen X, et al. Visual prompt multi-modal tracking. In *CVPR* 2023: 9516-9526.

[2] Wang X, Li J, Zhu L, et al. VisEvent: Reliable object tracking via collaboration of frame and event flows. *IEEE T-CYB*, 2023, 54(3):1997-2010.

[3] Tang C, Wang X, Huang J, et al. Revisiting color-event based tracking: A unified network, dataset, and metric. *Pattern Recognition*, 2025, 7:112718.

## 许可

本项目仅用于学术与研究目的。
