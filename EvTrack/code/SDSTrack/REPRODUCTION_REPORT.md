# SDSTrack 复现报告（Issue #4）

> **关联 Issue:** [#4 SDSTrack复现](https://github.com/kriss-spy/EvTrack/issues/4)  
> **负责人:** @kriss-spy (Wang Di)  
> **目标:** 复现 [SDSTrack](https://github.com/hoqolo/SDSTrack) 并在 VisEvent/COESOT 数据集上进行训练与测试  

---

## 1. 项目背景与目标

### 1.1 课题要求

本课题（Topic #65：基于事件相机的目标跟踪）要求：
- 完成基于事件相机的目标跟踪相关文献调研并形成综述报告
- 重点研究极端光照或高速场景下的目标跟踪方法
- 探索事件与图像融合的跟踪框架设计
- **复现 ViPT 或 SDSTrack 算法并进行训练测试**
- 结合评价指标（精度、成功率及鲁棒性）对跟踪性能进行系统分析

### 1.2 SDSTrack 简介

**SDSTrack**（Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking）是由 Hou 等人发表于 CVPR 2024 的多模态视觉目标跟踪算法。

- **核心创新:** 自蒸馏对称适配器学习（Self-Distillation Symmetric Adapter），有效融合 RGB 帧与事件数据
- **基础架构:** 基于 OSTrack（Transformer-based tracker），引入轻量级对称适配器模块
- **应用场景:** 高速运动、极端光照等挑战性场景下的鲁棒目标跟踪

---

## 2. Issue #4 进展追踪

### 2.1 Issue 时间线

| 时间 | 事件 |
|------|------|
| 2026-05-04 | Issue #4 创建，链接上游仓库 |
| 2026-05-12 | 提交 Colab Notebook: `SDSTrack_VisEvent_eval.ipynb` |
| 2026-05-25 | **声明：训练复现不再计划**（training reproduction not planned） |
| 2026-05-29 | 打磨结果（polishing the result） |
| **2026-05-30** | **截止日，提交最终报告** |

### 2.2 关键决策

- **训练复现取消原因：**
  - 训练需要大量计算资源
  - 时间成本过高
  - **策略调整：** 使用上游预训练模型进行推理评估，专注于**评估指标复现**和**流程验证**

---

## 3. 复现环境与技术栈

### 3.1 实验环境（Google Colab）

<!-- 记不清了，用的可能比T4好 -->

| 组件 | 版本/规格 |
|------|----------|
| GPU | Tesla T4 |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Python | 3.12 |
| 显存 | 16 GB |

> ⚠️ **环境兼容性挑战：** 上游代码要求 PyTorch 1.11.0 + CUDA 11.3 + Python 3.8，Colab 已升级至 PyTorch 2.x + Python 3.12。

### 3.2 本地开发环境

- **路径:** `EvTrack/code/SDSTrack/`
- **Python:** 3.13
- **包管理:** `uv`
- **用途:** 仅用于编码和文档，不运行训练或推理

### 3.3 兼容性补丁

由于 PyTorch 2.x 和 Python 3.10+ 的破坏性变更，手动应用了以下补丁：

| 文件 | 问题 | 修复方案 |
|------|------|----------|
| `lib/train/data/loader.py` | `torch._six` 在 PyTorch 2.x 中已移除 | `try/except` 回退到 `(str, bytes)` 和 `int` |
| `lib/train/data/loader.py` | `collections.Mapping` / `Sequence` 在 Python 3.10+ 中已移除 | 替换为 `collections.abc.Mapping` / `collections.abc.Sequence` |
| `lib/test/tracker/sdstrack.py` | PyTorch 2.6+ 默认 `weights_only=True`，无法加载旧模型 | 显式添加 `weights_only=False` |
| `lib/train/trainers/base_trainer.py` | 同上 | 显式添加 `weights_only=False` |

补丁脚本：`scripts/patch_colab_compat.py`（已集成到 Colab Notebook 中自动应用）

---

## 4. 数据集准备

### 4.1 VisEvent 数据集

- **规模:** 232 GB（分卷压缩）
- **来源:** Dropbox 官方链接
- **存放位置:** Google Drive (`MyDrive/EvTrack/datasets/VisEvent/`)
- **结构:**
  ```
  data/visevent/
  ├── train/train_subset/     # 120 个序列
  └── test/test_subset/       # 77 个序列
  ```
- **每个序列包含:**
  - `vis_imgs/` — RGB 帧（.bmp）
  - `event_imgs/` — 事件帧（.bmp）
  - `groundtruth.txt` — 边界框标注
  - `absent_label.txt` — 目标缺失标签

### 4.2 COESOT 数据集

- **状态:** 尚未获取
- **困难:** 仅托管于百度网盘，无法直接从 Colab 下载
- **计划:** 需本地下载后上传至 Google Drive，或寻找其他镜像源

### 4.3 数据解压策略

使用 `7z` 直接解压分卷 zip（无需先合并，节省 ~130 GB 临时空间）：

```bash
apt-get install -y p7zip-full
7z x VisEvent_train.zip -o/content/drive/MyDrive/EvTrack/datasets/VisEvent/train
7z x VisEvent_test.zip -o/content/drive/MyDrive/EvTrack/datasets/VisEvent/test
```

---

## 5. 模型准备

### 5.1 预训练模型

- **OSTrack 基础模型:** `vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar`（354.2 MB）
  - 来源：Google Drive（官方提供）
  - 用途：SDSTrack 的训练初始化
- **SDSTrack 训练模型:** `SDSTrack_cvpr2024_rgbe.pth.tar`（489.0 MB）
  - 来源：Google Drive shortcut（绕过 Google 下载配额限制）
  - 用途：直接用于推理评估

### 5.2 模型获取方式

由于 Google Drive 共享文件夹存在下载配额限制，采用**快捷方式（shortcut）**策略：
1. 在共享文件夹中找到模型文件
2. 右键 → "整理" → "添加快捷方式" → "我的云端硬盘"
3. Colab 挂载后直接读取快捷方式指向的文件

---

## 6. 评估实验与结果

### 6.1 评估设置

- **测试脚本:** `RGBE_workspace/test_rgbe_mgpus.py`
- **配置:** `cvpr2024_rgbe.yaml`
- **测试集:** VisEvent test subset（77 个序列）
- **运行参数:**
  - GPU: 1 张 Tesla T4
  - 线程数: 1（为稳定性牺牲速度）
  - 预计用时: 30–90 分钟

### 6.2 评估结果

| 指标 | 数值 |
|------|------|
| **Success AUC** | **0.6238** |
| **Precision (20px)** | **0.7356** |
| 测试序列数 | 76 / 77 |
| 总帧数 | — |

### 6.3 结果说明

- **76/77 序列:** 1 个序列（`dvSave-2021_02_14_17_00_48`）因缺失 `vis_imgs` 目录而被跳过
- **指标含义:**
  - **Success AUC:** 不同 IoU 阈值下成功率曲线的曲线下面积，衡量整体跟踪成功率
  - **Precision (20px):** 中心位置误差小于 20 像素的帧比例，衡量定位精度

### 6.4 与官方对比

> 注：上游仓库未明确提供 VisEvent 数据集上的官方基准数值，需对照论文中的图表数据进行比较。

---

## 7. 技术难点与解决方案

### 7.1 环境断层问题

**问题:** 上游代码基于 PyTorch 1.11.0 + Python 3.8，Colab 已升级至 PyTorch 2.10.0 + Python 3.12，存在 API 不兼容。

**解决:**
- 自动检测并应用兼容性补丁（`torch._six`、`collections.abc`、`weights_only` 等）
- 编写幂等的 Colab Notebook 单元格，可重复运行不报错

### 7.2 路径硬编码问题

**问题:** 上游代码中数据集路径、模型路径写死为作者本地服务器路径。

**解决:**
- 自动重写 `lib/train/admin/local.py` 中的 `visevent_dir`
- 自动重写 `RGBE_workspace/test_rgbe_mgpus.py` 中的测试路径
- 使用符号链接（symlink）将 Google Drive 数据目录映射到工作区

### 7.3 数据集索引缺失

**问题:** 上游仓库缺少 `testlist.txt` 和 `trainlist.txt`，导致测试脚本无法运行。

**解决:** 在 Notebook 中自动扫描目录生成索引文件：
```python
seqs = sorted([d for d in os.listdir(dirname) if os.path.isdir(...)])
```

### 7.4 Colab 长时间运行稳定性

**问题:** 评估 77 个序列需 30–90 分钟，Colab 可能因空闲而断开连接。

**解决:**
- 内置 keepalive 线程（每 30 秒打印进度）
- 支持断点续跑（已完成的序列自动跳过）
- 单线程运行以提高稳定性

---

## 8. 项目文件结构

```
EvTrack/code/SDSTrack/
├── README.md                          # 项目说明
├── PLAN.md                            # 详细复现计划
├── pyproject.toml                     # uv 项目配置
├── .python-version                    # Python 3.13
├── main.py                            # 占位入口文件
├── SDSTrack_VisEvent_eval.ipynb       # Colab 评估笔记本（核心）
├── uv.lock                            # uv 依赖锁定
├── scripts/
│   └── patch_colab_compat.py          # 兼容性补丁脚本
└── docs/
    └── dataset-mount.md               # 数据集下载指南
```

---

## 9. 复现流程总结

### 9.1 复现步骤

1. **打开 Colab Notebook:** `SDSTrack_VisEvent_eval.ipynb`
2. **Phase 1 — 环境准备:**
   - 挂载 Google Drive或调用dropbox API接入dropbox上的数据集
   - 验证 GPU / CUDA
   - 安装依赖包
   - 克隆 SDSTrack 上游仓库
   - 应用兼容性补丁
3. **Phase 2 — 数据准备:**
   - 创建符号链接指向 Google Drive 中的 VisEvent 数据集或以其他方式准备非数据集文件
   - 验证数据集完整性
4. **Phase 3 — 模型准备:**
   - 复制 OSTrack 预训练模型和 SDSTrack checkpoint
   - 应用 PyTorch 2.x 权重加载补丁
5. **Phase 4 — 评估:**
   - 生成 `testlist.txt` / `trainlist.txt`
   - 运行测试脚本（单线程，带 keepalive）
   - 自动计算 Success AUC 和 Precision @ 20px

### 9.2 Notebook 特性

- **所有单元格幂等** — 可重复运行，不会报错
- **自动状态检测** — 每个单元格先检查是否已完成
- **错误恢复友好** — 失败后可修复并重新运行
- **建议顺序运行** — 而非"运行全部"（数据集解压需数小时）

---

## 10. 当前状态与剩余工作

### 10.1 已完成

- [x] VisEvent 数据集下载（232 GB）
- [x] VisEvent 数据集解压（Train: 120 seqs, Test: 77 seqs）
- [x] 本地开发环境初始化（uv + Python 3.13）
- [x] SDSTrack 上游仓库克隆
- [x] Colab 环境配置（PyTorch 2.10 + T4 GPU）
- [x] OSTrack 预训练模型下载（354.2 MB）
- [x] SDSTrack 训练模型获取（489.0 MB）
- [x] 数据集索引文件生成（testlist.txt / trainlist.txt）
- [x] Colab Notebook 完成（所有单元格幂等）
- [x] **VisEvent 评估完成（76/77 seqs, AUC: 0.6238, Precision: 0.7356）**

### 10.2 待完成 / 阻塞

- [ ] **COESOT 数据集获取** — 需解决百度网盘下载问题
- [ ] COESOT 数据集评估
- [ ] 训练复现（已放弃，改用上游预训练模型）
- [ ] 与 ViPT（Issue #3）的对比实验
- [ ] 训练曲线、跟踪可视化结果
- [ ] 完整的实验文档（`docs/experiments.md`）

### 10.3 风险与应对

| 风险 | 状态 | 应对措施 |
|------|------|----------|
| Colab 断开导致训练中断 | 已规避 | 使用预训练模型，无需训练 |
| 显存不足 (OOM) | 未触发 | 单线程推理，batch size 默认为 1 |
| 数据集解压后路径不匹配 | 已解决 | 自动路径重写 + 符号链接 |
| 上游仓库依赖冲突 | 已解决 | 兼容性补丁脚本 |
| COESOT 数据集过大/下载慢 | **进行中** | 优先保证 VisEvent 实验完整，COESOT 作为补充 |

---

## 11. 结论

本次 SDSTrack 复现工作**成功完成了在 VisEvent 数据集上的评估**，获得了量化的跟踪性能指标（Success AUC: 0.6238, Precision: 0.7356），为课程设计报告提供了关键的实验数据支撑。

### 主要贡献

1. **环境兼容性方案:** 针对 PyTorch 2.x + Python 3.10+ 的破坏性变更，提供了一套完整的自动补丁方案
2. **Colab 自动化流程:** 设计了幂等、可断点续跑的 Notebook，大幅降低了复现门槛
3. **指标复现:** 在 VisEvent 测试集上成功运行了 SDSTrack 推理，获得了标准化评估指标

### 局限性

1. **未进行训练复现:** 仅使用上游预训练模型进行推理，未从头训练模型
2. **COESOT 数据集缺失:** 由于下载困难，尚未在 COESOT 上进行评估
3. **缺少对比实验:** 未与 ViPT 或其他基线方法进行直接对比

### 下一步建议

- 优先解决 COESOT 数据集获取问题，完成双数据集评估
- 与 ViPT 复现结果（Issue #3）进行对比分析
- 补充跟踪可视化结果（跟踪轨迹 overlay 视频）
- 撰写完整的实验分析章节，包含成功/失败案例分析

---

## 附录：关联资源

- **SDSTrack 论文:** Hou X, et al. SDSTrack: Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking. CVPR 2024.
- **SDSTrack 代码:** https://github.com/hoqolo/SDSTrack
- **VisEvent:** https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark
- **COESOT:** https://github.com/Event-AHU/COESOT
- **Colab Notebook:** `EvTrack/code/SDSTrack/SDSTrack_VisEvent_eval.ipynb`
- **本地代码:** `EvTrack/code/SDSTrack/`
- **GitHub Issue:** https://github.com/kriss-spy/EvTrack/issues/4
