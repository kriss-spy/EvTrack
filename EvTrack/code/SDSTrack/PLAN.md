# SDSTrack 复现计划 (Issue #4)

> **负责人:** @kriss-spy  
> **目标:** 复现 [SDSTrack](https://github.com/hoqolo/SDSTrack) 并在 VisEvent/COESOT 数据集上进行训练与测试  
> **关联 Issue:** [#4 SDSTrack复现](https://github.com/kriss-spy/pr-course-prj/issues/4)

---

## 1. 目标定义

完成以下任务：
1. 搭建 SDSTrack 运行环境；
2. 完成 VisEvent 数据集解压与预处理；
3. 运行 SDSTrack 的训练脚本（如时间/资源允许）；
4. 运行 SDSTrack 的测试/评估脚本，获取 **Precision**、**Success** 等指标；
5. 记录实验结果，形成可复现的实验流程文档。

---

## 2. 当前进度 (as of 2026-05-08)

| 任务 | 状态 | 备注 |
|------|------|------|
| VisEvent 数据集下载 | ✅ 完成 | 已下载至 Google Drive，共 232 GB |
| VisEvent 数据集解压 | ✅ 完成 | Train: 120 sequences, Test: 77 sequences |
| COESOT 数据集下载 | ⬜ 未开始 | **必须完成**。需解决 Baidu Netdisk 下载问题 |
| 本地开发环境 (uv) | ✅ 已初始化 | Python 3.13，用于编码和文档 |
| SDSTrack 上游仓库 | ✅ 已克隆 | `upstream/` 目录，已加入 `.gitignore` |
| Colab 环境配置 | ✅ 完成 | PyTorch 2.10 + T4 GPU，兼容性补丁已应用 |
| OSTrack 预训练模型 | ✅ 已下载 | 354.2 MB，用于训练初始化 |
| SDSTrack 训练模型 | ✅ 已获取 | 489.0 MB，通过 Google Drive shortcut |
| testlist.txt / trainlist.txt | ✅ 已创建 | 修复了缺失的数据集索引文件 |
| Colab 笔记本 | ✅ 完成 | 所有单元格幂等，含评估与指标计算 |
| **VisEvent 评估** | **✅ 完成** | **76/77 sequences, Success AUC: 0.6238, Precision: 0.7356** |
| **阶段总结** | **Phase 1 ✅, Phase 2 ✅, Phase 4 ✅ (VisEvent)** | **基线指标已获取，待 COESOT 数据集** |

---

## 3. 详细执行步骤

### Phase 1: 环境准备 (第 1–2 天)

#### 3.1.1 克隆 SDSTrack 上游仓库
```bash
cd EvTrack/code/SDSTrack
git clone https://github.com/hoqolo/SDSTrack.git upstream
```
> **注意:** 上游代码加入 `.gitignore`，避免提交 vendor 代码。

#### 3.1.2 分析依赖
- 阅读 `upstream/README.md` 与 `requirements.txt` / `setup.py` / `install.sh`；
- 记录 PyTorch 版本、CUDA 版本、特殊依赖（如 `libtorch`, `opencv-python`, `pysot`, `got10k` 等）。

**上游依赖分析结果 (2026-05-05):**
- Python: 3.8（上游建议）
- PyTorch: 1.11.0 + CUDA 11.3 + torchvision 0.12.0
- 关键包: PyYAML, easydict, cython, opencv-python, pandas, pycocotools, jpeg4py, scipy, timm==0.5.4, tb-nightly, lmdb, visdom, wandb, vot-toolkit==0.5.3, vot-trax==3.0.3
- ⚠️ **CRITICAL COMPATIBILITY ISSUE:** Colab currently ships Python 3.12 + CUDA 12.8. PyTorch 1.11.0 wheels are **not available** for Python 3.12. We must use Colab's pre-installed PyTorch 2.10.0+cu128. This may cause upstream code breakage.
- 需下载 OSTrack 预训练权重：`pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar`（Google Drive/Baidu Netdisk）

#### 3.1.3 配置环境（本地编码 + Colab 实验）
- **本地编码/文档:** 使用 `uv` 管理依赖，更新 `pyproject.toml`，用 `uv add` 安装依赖。不用于运行训练或推理。
- **Colab 实验:** 所有训练、推理、数据解压等 heavy compute 均在 **`SDSTrack.ipynb`**（通过 Colab MCP 连接）中执行。Colab 中如需 Conda 环境，优先使用 `mamba`，不要用 `conda`。

#### 3.1.4 Colab 笔记本特性
- **所有单元格均为幂等的（idempotent）** — 可以重复运行，不会报错
- **自动状态检测** — 每个单元格会先检查是否已完成，已完成则跳过
- **错误恢复友好** — 如果某个单元格失败，修复后重新运行即可
- **建议始终使用「顺序运行」而非「运行全部」**，因为数据集解压需要数小时

#### 3.1.5 验证环境
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

---

### Phase 2: 数据集准备 (第 2–4 天)

#### 3.2.1 解压 VisEvent
数据集当前为分卷压缩格式（`.zip` + `.z01`–`.z06`），**在 Colab 中解压**（不要在本地笔记本上执行）：

```bash
# 在 Colab / Google Drive 中执行
# 使用 7z 直接解压分卷 zip，无需先合并，节省临时空间
cd /content/drive/MyDrive/EvTrack/datasets/VisEvent/VisEvent_dataset
apt-get install -y p7zip-full
7z x VisEvent_train.zip -o/content/drive/MyDrive/EvTrack/datasets/VisEvent/train
7z x VisEvent_test.zip -o/content/drive/MyDrive/EvTrack/datasets/VisEvent/test
```

**为什么用 7z 而不是 `zip -s-` + `unzip`？**
- 无需先合并分卷（节省 ~130 GB 临时空间）
- 单遍读写，速度更快
- 原生支持 `.z01`–`.z06` 分卷格式

#### 3.2.2 数据集目录结构对齐

**实际解压后的结构：**
```
data/visevent/train/
└── train_subset/           # zip 中的根文件夹
    ├── 00142_tank_outdoor2/
    │   ├── vis_imgs/       # RGB frames (.bmp)
    │   ├── event_imgs/     # Event frames (.bmp)
    │   ├── groundtruth.txt # bbox annotations
    │   └── absent_label.txt
    ├── ... (120 sequences)
    └── trainlist.txt

data/visevent/test/
└── test_subset/            # zip 中的根文件夹
    ├── 00141_tank_outdoor2/
    │   ├── vis_imgs/
    │   ├── event_imgs/
    │   ├── groundtruth.txt
    │   └── absent_label.txt
    ├── ... (77 sequences)
    └── testlist.txt
```

**路径适配：**
- 上游 `lib/train/admin/local.py` 中的 `visevent_dir` 已自动修改为 `.../train/train_subset`
- 上游 `RGBE_workspace/test_rgbe_mgpus.py` 中的硬编码测试路径已自动修改为 `.../test/test_subset`

#### 3.2.3 生成数据集索引
- 如上游仓库提供数据加载脚本，直接复用；
- 否则，编写 `tools/prepare_visevent.py`，生成符合 SDSTrack 格式的 `dataset.json` 或 `.txt` 列表。

#### 3.2.4 COESOT 数据集 (必须)
- **COESOT 是必需数据集**，必须在两个数据集上都进行实验。
- COESOT 仅托管在 **Baidu Netdisk**，无法直接用 Colab 下载（不像 VisEvent 可通过 Dropbox API）。
- **获取策略（待定）：**
  - 方案 A：本地笔记本下载后上传至 Google Drive，再挂载到 Colab；
  - 方案 B：寻找是否有其他镜像源（如 Hugging Face、学校服务器）；
  - 方案 C：联系 COESOT 作者获取其他下载方式。
- 需在 Phase 2 早期解决此问题，避免阻塞后续实验。

---

### Phase 3: 模型训练 (第 4–10 天，视算力而定)

#### 3.3.1 预训练模型
- **必须下载 OSTrack 基础模型**（foundation model）：`vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar`
  - 来源: [Google Drive](https://drive.google.com/drive/folders/1ttafo0O5S9DXK2PX0YqPvPrQ-HWJjhSy?usp=sharing) 或上游 README 中的 Baidu Netdisk 链接
  - 放置路径: `./pretrained/vitb_256_mae_ce_32x4_ep300/OSTrack_ep0300.pth.tar`
- SDSTrack 自身 checkpoint: 上游提供训练好的模型（Google Drive / Baidu Netdisk: `qolo`），可直接用于测试；如需复现训练过程则从头训练。

#### 3.3.2 训练脚本适配
- 复制上游训练脚本到本项目目录（如 `scripts/train_sdstrack.sh`）；
- 修改配置文件中的数据路径、batch size、GPU 数量，使其适配 **Colab** 环境（T4/A100）。

#### 3.3.3 Colab 训练策略
- 使用 Colab Pro / T4 / A100 GPU；
- 设置 checkpoint 保存到 Google Drive，防止断连丢失；
- 建议每 epoch 保存一次，并记录 `train.log`。

#### 3.3.4 训练监控
- 使用 `tensorboard` 或 `wandb` 记录 loss、lr 曲线；
- 监控 GPU 利用率，确保无数据加载瓶颈。

> **时间评估:** 若从头训练，基于事件的多模态跟踪器通常需要 1–3 天（取决于 GPU 与 epoch 数）；如仅做推理测试，可跳过此阶段。

---

### Phase 4: 测试与评估 (第 3–5 天，可与训练并行规划)

#### 3.4.1 测试脚本运行
- 上游测试脚本：`RGBE_workspace/test_rgbe_mgpus.py`
- 修改硬编码的 `seq_home` 路径为 Google Drive 中的 VisEvent 测试集路径；
- 运行命令：`bash eval_rgbe.sh`
- 生成结果：每序列一个 `.txt` 文件，保存于 `./RGBE_workspace/results/VisEvent/cvpr2024_rgbe/`

#### 3.4.2 评估指标计算
- **Precision Plot:** 中心位置误差阈值下的精度；
- **Success Plot:** 重叠率阈值下的成功率；
- **AUC / OP:** 曲线下面积与平均重叠率；
- **速度:** FPS (frames/events per second)。

使用官方评估工具或 `pysot` 工具包：
```bash
python tools/eval.py --dataset VisEvent --tracker_result results/SDSTrack_VisEvent_test/
```

#### 3.4.3 对比实验
- 如可能，与 ViPT (Issue #3) 或基线方法（如 OSTrack、TransT）进行对比；
- 在相同测试子集上运行，保证公平性。

---

### Phase 5: 结果整理与文档化 (第 5–7 天)

#### 3.5.1 实验记录
- `EvTrack/code/SDSTrack/docs/experiments.md`：记录每次实验的超参数、环境、结果；
- 保存训练曲线截图、测试可视化结果（如跟踪轨迹 overlay）。

#### 3.5.2 指标汇总表
| 数据集 | 方法 | Precision (20px) | Success AUC | FPS | 备注 |
|--------|------|------------------|-------------|-----|------|
| VisEvent | SDSTrack (upstream model) | 0.7356 | 0.6238 | — | 76/77 sequences (1 incomplete: `dvSave-2021_02_14_17_00_48` missing `vis_imgs`) |
| VisEvent | SDSTrack (our training) | — | — | — | 待训练 |
| COESOT | SDSTrack | — | — | — | 待 COESOT 数据集 |

#### 3.5.3 可复现性文档
- 更新 `EvTrack/code/SDSTrack/README.md`，包含：
  - 环境安装命令；
  - 数据准备步骤；
  - 训练/测试一键运行命令；
  - 已知问题与解决方法。

---

## 4. 风险与应对

| 风险 | 可能性 | 应对措施 |
|------|--------|----------|
| Colab 断开导致训练中断 | 高 | 使用 Google Drive 保存 checkpoint；缩短 epoch 间隔 |
| 显存不足 (OOM) | 中 | 减小 batch size、使用混合精度 (`amp`)、减少输入分辨率 |
| 数据集解压后路径不匹配 | 中 | 仔细阅读上游代码的数据加载部分，必要时重写 `dataset.py` |
| 上游仓库依赖冲突 | 中 | 使用 `mamba` 创建隔离环境（不用 `conda`）；记录可用的版本组合 |
| COESOT 数据集过大/下载慢 | 低 | 优先保证 VisEvent 实验完整，COESOT 作为加分项 |

---

## 5. 兼容性记录 (Compatibility Notes)

### PyTorch 2.x + Python 3.10+ 适配

上游代码基于 PyTorch 1.11.0 + Python 3.8 编写。Colab 当前环境为 PyTorch 2.10.0 + Python 3.12，需应用以下补丁：

| 文件 | 问题 | 修复 |
|------|------|------|
| `lib/train/data/loader.py` | `torch._six` 在 PyTorch 2.x 中已移除 | 用 `try/except` 回退到 `(str, bytes)` 和 `int` |
| `lib/train/data/loader.py` | `collections.Mapping` / `collections.Sequence` 在 Python 3.10+ 中已移除 | 替换为 `collections.abc.Mapping` / `collections.abc.Sequence` |

这些补丁已集成到 `SDSTrack.ipynb` 的 setup cell 中，克隆上游仓库后自动应用。本地 `scripts/patch_colab_compat.py` 也可用于手动打补丁。

---

## 6. 时间线 (参考课程设计指导书)

| 周次 | 任务 | 交付物 |
|------|------|--------|
| 第 12 周 | 方案设计/修改 | 本计划文档 |
| 第 13–14 周 | 环境搭建 + 数据准备 | 可运行的 baseline 环境 |
| 第 15 周 | 训练模型 | checkpoint 文件 + 训练日志 |
| 第 16–17 周 | 测试评估 + 对比实验 | 指标表格 + 结果可视化 |
| 第 18 周 | 撰写报告 | 实验章节 + 可复现说明 |

---

## 7. 关联资源

- **SDSTrack 论文:** Hou X, et al. SDSTrack: Self-Distillation Symmetric Adapter Learning for Multi-Modal Visual Object Tracking. CVPR 2024.
- **SDSTrack 代码:** https://github.com/hoqolo/SDSTrack
- **VisEvent:** https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark
- **COESOT:** https://github.com/Event-AHU/COESOT
- **Colab 数据集路径:** `/content/drive/MyDrive/EvTrack/datasets/VisEvent/`
