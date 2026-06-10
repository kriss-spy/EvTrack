# ViPT 量化探索方案

> 对应 GitHub Issue #9：对 ViPT 的权重量化，降低数值精度，评估对预测精度等评价指标的影响。

---

## 1. 项目与模型理解

### 1.1 EvTrack 项目结构

EvTrack 旨在复现并评估 **ViPT (Visual Prompt Multi-Modal Tracking)** 在事件相机 (Event Camera) 跟踪基准上的性能。

当前项目包含：
- `code/patches/`：对原始 ViPT 的改进补丁
  - `ostrack_prompt.py`：模型构建（`build_viptrack`）
  - `vit_ce_prompt.py`：带 Candidate Elimination (CE) 的 Vision Transformer
  - `vipt_online_template.py` / `vipt_single_template.py`：跟踪器逻辑（在线模板更新）
- 数据集：VisEvent、COESOT（RGB + Event 双模态）

### 1.2 ViPT 模型架构特点

ViPT 的核心架构基于 **Vision Transformer (ViT)**，关键特征：

| 组件 | 说明 | 参数量/计算特征 |
|------|------|----------------|
| **Backbone** | `vit_base_patch16_224_ce_prompt` (ViT-B/16) | ~86M 参数 (OSTrack 预训练) |
| **Prompt Blocks** | 浅层/深层 prompt 分支（`vipt_shaw` / `vipt_deep`） | 0.84M 可训练参数 |
| **CEBlock** | Candidate Elimination 注意力块 | 稀疏化搜索区域 token |
| **Box Head** | Corner/Center Head | 输出目标框坐标 |
| **Patch Embed** | 两组：RGB 用 `patch_embed`，Event 用 `patch_embed_prompt` | Conv2d 层 |

**关键观察**：
- ViPT 的绝大部分参数量在 **Transformer Blocks (Linear / QKV / MLP)** 和 **Patch Embedding (Conv2d)** 中。
- Prompt 部分参数量极小（<1%），但负责跨模态融合，精度敏感。
- 模型使用 **FP32** 权重加载（`torch.load(..., map_location='cpu')`），推理时转入 CUDA。

---

## 2. 神经网络量化基础

### 2.1 为什么要量化？

- **存储压缩**：FP32 → INT8 可将模型体积压缩为 **1/4**。
- **推理加速**：INT8 矩阵乘法在支持低精度指令的硬件（如 NVIDIA Tensor Core、ARM NEON）上有显著加速。
- **带宽降低**：显存/内存带宽需求减少，缓存效率提升。
- **事件相机场景**：嵌入式边缘设备（如 Prophesee 事件相机处理平台）往往算力受限，低精度推理更易部署。

### 2.2 量化核心公式

均匀线性量化（Uniform Affine Quantization）：

$$x_q = \text{clamp}\left(\text{round}\left(\frac{x}{s} + z\right), q_{min}, q_{max}\right)$$

$$x_{dq} = s \cdot (x_q - z)$$

其中：
- $s$ (scale)：浮点缩放因子
- $z$ (zero_point)：零点偏移（对称量化时 $z=0$）
- $q_{min}, q_{max}$：INT8 时通常为 $-128, 127$（非对称）或 $-127, 127$（对称）

### 2.3 三种主要量化范式

| 范式 | 名称 | 权重 | 激活 | 是否需要校准/训练 | 精度损失 | 适用场景 |
|------|------|------|------|------------------|----------|----------|
| **PTQ** | Post-Training Quantization (后训练量化) | INT8 | 静态 INT8 | 需要校准集（~100-500张） | 中低 | 最快部署，无训练资源 |
| **QAT** | Quantization-Aware Training (量化感知训练) | INT8 | 伪量化 FP32 | 需要微调训练 | 最低 | 精度敏感，有训练资源 |
| **Dynamic** | Dynamic Quantization (动态量化) | INT8 | 运行时动态 | 无需校准 | 中 | 仅 CPU 推理，RNN/LSTM |

### 2.4 针对 Vision Transformer 的难点

ViT 的量化比 CNN 更难，原因包括：
1. **Post-Softmax 激活分布**：注意力权重呈幂律分布，极端值多。
2. **Post-GELU 激活**：有负值，动态范围大。
3. **LayerNorm 输入**：通道间分布差异大，容易受异常值 (outlier) 影响。
4. **残差连接**：量化误差会在深层累积。

近年来 SOTA 的 ViT 量化方法（如 **RepQ-ViT**, **APHQ-ViT**, **AdaLog**, **QwT**, **GPLQ**）主要解决上述问题。

---

## 3. ViPT 量化实施路线

### 3.1 推荐路线：渐进式量化实验

对于课程项目/研究探索，建议采用 **由浅入深、逐步评估** 的策略：

```
Phase 1: FP16/BF16 混合精度（零成本尝试）
    ↓ 评估速度增益、精度变化
Phase 2: 权重量化 (Weight-Only, INT8) — 动态量化
    ↓ 评估对 tracking 精度的影响
Phase 3: 静态后训练量化 (PTQ, W8A8) — 需要校准集
    ↓ 评估速度、精度
Phase 4 (可选): 量化感知训练 (QAT) — 如有 GPU 训练资源
    ↓ 恢复精度
Phase 5 (可选): 低比特量化 (INT4/FP4) — 如硬件支持
```

### 3.2 各阶段具体方案

---

#### Phase 1: FP16 / BF16 混合精度推理

**方法**：最简单，直接将模型转为 `half()` 精度。

```python
model = model.half()  # 或 model.to(torch.bfloat16)
# 输入也需要转 half
template = template.half()
search = search.half()
```

**效果**：
- 模型体积减半。
- 现代 GPU (Tensor Core) 可加速 FP16 运算。
- **对跟踪精度通常几乎无影响**（因为 ViPT 的 Prompt 部分仍可用 FP32）。

**适用性**：快速验证，作为后续 INT8 实验的 baseline。

---

#### Phase 2: 动态权重量化 (Dynamic Quantization, Weight-Only INT8)

**方法**：PyTorch 原生 `torch.quantization.quantize_dynamic`

仅将 **Linear 层** 的权重静态量化到 INT8，激活值仍为 FP32（运行时动态反量化）。

```python
import torch
from torch.quantization import quantize_dynamic

# 加载 ViPT 模型
network = build_viptrack(cfg, training=False)
network.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['net'], strict=True)

# 仅对 Linear 层进行动态量化
quantized_network = quantize_dynamic(
    network,
    qconfig_spec={torch.nn.Linear},  # 只量化 Linear
    dtype=torch.qint8
)

quantized_network = quantized_network.cuda().eval()
```

**特点**：
- 无需校准数据，一键完成。
- 仅减少模型加载时的内存占用，**实际 CUDA 加速有限**（因为激活仍是 FP32，且 PyTorch 动态量化主要优化 CPU 上的 GEMM）。
- 对精度影响通常很小（因为权重回退到 FP32 计算）。

---

#### Phase 3: 静态后训练量化 (PTQ, W8A8)

这是最有研究价值的路线。需要代表性校准数据（如 VisEvent 验证集的前 N 个序列）。

**挑战**：PyTorch Eager Mode 的量化对 Transformer 支持有限，且 ViPT 包含自定义 `CEBlock`、`Prompt_block`、`VisionTransformerCE`，直接使用 `torch.quantization.prepare` 可能遇到算子不支持的问题。

**推荐工具链**：

| 工具 | 适用场景 | 说明 |
|------|----------|------|
| **PyTorch FX Graph Mode** | 通用 PTQ | `torch.ao.quantization.fx.prepare_fx`，可处理自定义模块，但需配置 `qconfig_mapping` |
| **torchao** (PyTorch AO) | 现代推荐 | `torchao.quantization.quantize_`，支持 `Int8DynamicActivationInt8WeightConfig` 等，对 Linear 层友好 |
| **NVIDIA ModelOpt + TensorRT** | 生产级 GPU 加速 | `mtq.quantize()` + Torch-TensorRT，适合最终部署 |
| **RepQ-ViT / QwT / APHQ-ViT** | 研究级 ViT 量化 | 针对 ViT 的 SOTA 开源方案，精度恢复好 |

##### 方案 A：使用 `torchao` 对 Linear 层做快速 INT8 PTQ (推荐入门)

```python
import torch
from torchao.quantization import quantize_, Int8DynamicActivationInt8WeightConfig

# 1. 加载模型
network = build_viptrack(cfg, training=False).cuda().eval()

# 2. 校准：用少量数据跑前向，收集激活分布
# 需要一个 calibration_loop
def calibration_loop(model, dataloader, num_batches=10):
    model.eval()
    with torch.no_grad():
        for i, (template, search, mask) in enumerate(dataloader):
            if i >= num_batches:
                break
            _ = model(template, search, ce_template_mask=mask)

# 3. 量化
quantize_(network, Int8DynamicActivationInt8WeightConfig())

# 4. 编译加速（可选，需 PyTorch 2.x）
# network = torch.compile(network, mode='max-autotune')
```

> ⚠️ 注意：`torchao` 主要作用于 `nn.Linear`。对于 ViPT 中的 `nn.Conv2d`（PatchEmbed、Prompt_block）、`nn.MultiheadAttention`（CEBlock 内部），需要额外处理。

##### 方案 B：使用 PyTorch FX Graph Mode（更精细控制）

```python
from torch.ao.quantization import get_default_qconfig, prepare_fx, convert_fx
from torch.ao.quantization.quantize_fx import QuantizationConfig

# 配置 QConfig
qconfig = get_default_qconfig('x86')  # 或 'qnnpack' for ARM
qconfig_mapping = QConfigMapping().set_global(qconfig)

# 准备模型
example_inputs = (template, search, mask)  # 示例输入
prepared_model = prepare_fx(network, qconfig_mapping, example_inputs)

# 校准
with torch.no_grad():
    for data in calib_loader:
        prepared_model(*data)

# 转换
quantized_model = convert_fx(prepared_model)
```

**难点**：
- `VisionTransformerCE` 中的 `candidate_elimination_prompt`、`token2feature`/`feature2token` 等自定义操作可能不直接支持 FX 量化图。
- 可能需要将某些模块加入 `skip_quantize` 列表。

##### 方案 C：使用 NVIDIA ModelOpt（GPU 最强性能路线）

```python
import modelopt.torch.quantization as mtq

# 使用默认 INT8 配置
quant_cfg = mtq.INT8_DEFAULT_CFG

# 量化
mtq.quantize(network, quant_cfg, forward_loop=calibration_loop)

# 导出为 TensorRT
import torch_tensorrt
trt_model = torch_tensorrt.compile(
    network,
    ir="dynamo",
    arg_inputs=example_inputs,
    min_block_size=1,
    # 自动识别 QDQ 节点并融合为 INT8 kernel
)
```

---

#### Phase 4 (可选): 量化感知训练 (QAT)

如果 PTQ 导致精度下降过大（如 Success Score 下降 > 3%），可采用 QAT。

由于 ViPT 原始训练需要 A100 和大量数据，完整重训练成本太高。建议采用 **轻量 QAT**：

1. **冻结 Backbone 大部分参数**（仅保留 Prompt 和 Box Head 可训练）。
2. **插入 FakeQuantize 节点**（模拟 INT8 前向/反向）。
3. **在 VisEvent 训练集子集上微调 1-3 个 epoch**。

```python
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat_fx

# 准备 QAT
qconfig = get_default_qat_qconfig('x86')
model_prepared = prepare_qat_fx(network, qconfig_mapping, example_inputs)

# 训练（微调）
optimizer = torch.optim.AdamW(
    [p for n, p in model_prepared.named_parameters() if 'prompt' in n or 'box_head' in n],
    lr=1e-5
)
for epoch in range(3):
    for data in train_loader:
        ...
        loss.backward()
        optimizer.step()

# 转换为量化模型
model_quantized = convert_fx(model_prepared)
```

---

#### Phase 5 (进阶): 针对 ViPT 架构的混合精度策略

不是所有层都同样敏感。可研究 **分层量化**（Layer-wise Quantization）：

| 层/模块 | 敏感度 | 建议量化精度 |
|---------|--------|-------------|
| `patch_embed` (RGB) | 中 | INT8 |
| `patch_embed_prompt` (Event) | **高**（事件数据稀疏） | **FP16** 或保留 FP32 |
| `Prompt_block` (模态融合) | **极高** | **FP16** 或保留 FP32 |
| `blocks` (Transformer 前段 1-6) | 低 | INT8 |
| `blocks` (Transformer 后段 7-12) | 中 | INT8 |
| `box_head` | 高 | FP16 / INT8 |

> 研究问题：事件模态的稀疏性是否使得 Event 分支的权重对量化更敏感？

---

## 4. 评估指标与实验设计

### 4.1 需要评估的指标

| 指标 | 说明 | 量化前后对比 |
|------|------|-------------|
| **Success Score** | IoU > 阈值的帧比例 | 核心跟踪精度 |
| **Precision Score** | 中心位置误差 < 阈值的帧比例 | 核心跟踪精度 |
| **Normalized Precision** | 归一化精度 | 核心跟踪精度 |
| **FPS** | 推理帧率 | 速度提升 |
| **Model Size** | 权重文件大小 (MB) | 压缩率 |
| **FLOPs / MACs** | 理论计算量 | 硬件效率 |
| **GPU Memory** | 推理峰值显存 | 部署效率 |

### 4.2 实验基线

在 **VisEvent** 和 **COESOT** 上跑以下配置：

1. **Baseline FP32**：原始 `vipt_online_template.py` 的精度与速度。
2. **FP16 Baseline**：`model.half()` 的精度与速度。
3. **PTQ W8A8 (torchao)**：全模型 INT8 的精度与速度。
4. **PTQ W8A8 (分层混合)**：Event/Prompt 保留 FP16，其余 INT8。
5. (可选) **QAT W8A8**：微调后的精度与速度。

### 4.3 校准数据准备

```python
# 从 VisEvent 验证集中选取少量序列
# 每个序列取前 30 帧，构建 calibration dataset
calib_sequences = ['video0001', 'video0005', 'video0010', ...]  # 约 10 个序列

# 生成 (template, search, mask) 三元组
# 注意：保持与测试时相同的预处理（PreprocessorMM）
```

---

## 5. 风险与难点

1. **算子支持**：PyTorch 量化图模式对 `LayerNorm`、`GELU`、`Softmax` 的量化支持有限。ViT 中大量使用这些算子。
2. **Custom CEBlock**：`candidate_elimination_prompt` 中的动态 token 索引操作可能无法直接量化。
3. **事件模态特殊性**：Event 数据经过 `patch_embed_prompt` 后分布与 RGB 不同，统一量化参数可能次优。
4. **精度-速度权衡**：PTQ 可能精度掉点明显；QAT 需要训练资源。
5. **硬件依赖性**：真正的 INT8 加速需要底层库支持（TensorRT、ONNX Runtime、OpenVINO）。纯 PyTorch eager mode 的 INT8 推理在 GPU 上不一定比 FP16 快。

---

## 6. 推荐代码实现策略

### 6.1 最小可行验证 (MVP)

建议先用 **Phase 1 (FP16)** 和 **Phase 2 (Dynamic Weight-Only)** 快速验证，因为它们几乎无成本：

```python
# 在 vipt_online_template.py 的 __init__ 中尝试：

# 1. FP16 快速验证
# self.network = network.cuda().half()

# 2. 动态权重量化
from torch.quantization import quantize_dynamic
self.network = quantize_dynamic(network, {torch.nn.Linear}, dtype=torch.qint8).cuda()
```

然后直接跑 `test_rgbe_mgpus.py` 看精度变化。

### 6.2 下一步代码实现

如需要完整实现，建议创建 `code/quantization/` 目录：

```
code/quantization/
├── calibrate_vipt.py       # 校准数据生成与预处理
├── ptq_torchao.py          # torchao PTQ 入口
├── ptq_fx.py               # PyTorch FX Graph PTQ 入口
├── evaluate_quantized.py   # 量化模型评估脚本
├── utils.py                # 量化辅助函数（分层量化、跳过敏感模块）
└── README.md               # 使用说明
```

---

## 7. 参考文献与工具

### 7.1 论文

- **ViPT (CVPR 2023)**: Zhu et al. *Visual Prompt Multi-Modal Tracking.*
- **RepQ-ViT**: Li et al. *RepQ-ViT: Scale Reparameterization for Post-Training Quantization of Vision Transformers.*
- **APHQ-ViT (CVPR 2025)**: Wu et al. *Post-Training Quantization with Average Perturbation Hessian Based Reconstruction.*
- **AdaLog**: Wu et al. *AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer.*
- **QwT (CVPR 2025)**: Wu et al. *Quantization without Tears.*
- **GPLQ (NeurIPS 2025)**: Wu et al. *GPLQ: A General, Practical, and Lightning QAT Method for Vision Transformers.*
- **PowerYOLO**: Mixed-precision event-based detection (INT8 + LOG4).

### 7.2 工具链

- PyTorch Quantization: https://pytorch.org/docs/stable/quantization.html
- torchao (新一代量化 API): https://github.com/pytorch/ao
- NVIDIA ModelOpt: https://github.com/NVIDIA/TensorRT-Model-Optimizer
- RepQ-ViT / QwT / APHQ-ViT 官方实现（GitHub 开源）

---

## 8. 结论

对 ViPT 进行量化是**可行且有研究价值**的方向：

- **最低成本**：FP16 / BF16 混合精度（几乎零精度损失，体积减半）。
- **推荐路线**：使用 `torchao` 做 **Linear 层 INT8 PTQ**，配合少量 VisEvent 校准数据，评估对 Success/Precision 的影响。
- **进阶方向**：
  - 探索 **分层混合精度**（Event 分支保留 FP16，RGB/Transformer 用 INT8）。
  - 如 PTQ 掉点明显，尝试 **轻量 QAT**（仅微调 Prompt + Head）。
  - 最终可结合 **TensorRT** 实现真正的硬件加速。

核心科学问题：**事件模态分支的稀疏性是否使其对量化更敏感？** 这是可以写入课程设计报告的创新点。
