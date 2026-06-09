# EvTrack

## 65 基于事件相机的目标跟踪

### 问题描述

在复杂动态场景中，目标跟踪是视觉感知系统中的核心问题之一。传统基于帧相机的目标跟踪方法依赖固定帧率采样与积分成像机制，易受到运动模糊、遮挡干扰以及光照剧烈变化的影响，难以在高速运动或高动态范围条件下保持稳定性能。同时，帧式数据存在时间冗余与信息延迟的问题，不利于实时性要求较高的应用场景。相比之下，事件相机采用像素级异步触发机制，仅在亮度发生变化时输出事件流，具备微秒级时间分辨率与高动态范围特性，为目标的连续时空建模提供了全新的数据表达形式。因此，如何充分挖掘事件数据的时空稀疏结构，并与传统图像信息进行有效融合，成为提升目标跟踪鲁棒性与实时性的关键问题。本课题要求**完成基于事件相机的目标跟踪相关文献调研并形成综述报告，重点研究极端光照或高速场景下的目标跟踪方法，探索事件与图像融合的跟踪框架设计，对 [ViPT](https://github.com/jiawen-zhu/ViPT)[1]或 [SDSTrack](https://github.com/hoqolo/SDSTrack)[2]算法进行复现与训练测试，并结合评价指标（如精度、成功率及鲁棒性）对跟踪性能进行系统分析**。

### 数据描述

[VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)[4]

[COESOT](https://github.com/Event-AHU/COESOT)[5]

### 参考文献

[1] Zhu J, Lai S, Chen X, et al. Visual prompt multi-modal tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 2023: 9516-9526.
[2] Hou X, Xing J, Qian Y, et al. Sdstrack: Self-distillation symmetric adapter learning for multi-modal visual object tracking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition 2024: 26551-26561.
[3] Sun C., Zhang J., Wang Y., et al. Exploring Historical Information for RGBE Visual
Tracking with Mamba. IEEE/CVF Conference on Computer Vision and Pattern Recognition,
2025: 6500-6509.
[4] Wang X, Li J, Zhu L, et al. Visevent: Reliable object tracking via collaboration of frame and event flows. IEEE transactions on cybernetics. 2023, 54(3):1997-2010.
[5] Tang C, Wang X, Huang J, et al. Revisiting color-event based tracking: A unified network, dataset, and metric. Pattern Recognition. 2025, 7:112718.

### 指导老师

Yi Chang

## 分工 (2026-05-04)

- programmer 1 ViPT: CYJ
- programmer 2 SDSTrack: @kriss-spy
- researcher 1: @melioll
- researcher 3: @zhangrongxuan
- researcher 2: JTY

## 时间表

见[指导书 三、课程具体步骤与时间安排](../《模式识别》课程设计指导书（2026）修订版1.pdf)

1. 课程设计任务书的布置,讲解(0.5天)
2. 学生根据任务书的要求初步进行需求分析(第10周)
3. 进行方案设计,并撰写设计方案(第11周)
4. 指导老师审阅方案设计报告,根据意见修改设计方案(第12周)
5. 算法设计与实现(第13-15周)
6. 软件实现与算法测试(第16-17周)
7. 撰写课程设计报告(第18周)
8. 考核答辩(第19周)
