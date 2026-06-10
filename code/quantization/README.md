# ViPT 量化实验代码

本目录包含对 ViPT 进行权重量化/激活量化的实验脚本。

对应 GitHub Issue: [#9 vipt量化探索](https://github.com/kriss-spy/EvTrack/issues/9)

---

## 前置条件

1. 确保已初始化 `vipt` submodule：
   ```bash
   cd /path/to/EvTrack
   git submodule update --init
   ```

2. 确保 ViPT 依赖已安装（参考 `code/vipt/requirements.txt`）。

3. （可选）如需使用 `torchao` 量化：
   ```bash
   pip install torchao
   ```

---

## 快速开始

### 1. 零成本验证：FP16 / Dynamic INT8

修改 `vipt_online_template.py` 的 `__init__` 中模型加载部分：

```python
# 在 self.network = network.cuda() 之后添加：

# 方案 A: FP16
self.network = self.network.half()

# 方案 B: Dynamic INT8 (仅权重量化，CPU 优化为主)
from torch.quantization import quantize_dynamic
self.network = quantize_dynamic(self.network, {torch.nn.Linear}, dtype=torch.qint8).cuda()
```

然后正常运行 `test_rgbe_mgpus.py` 评估即可。

### 2. 使用 `torchao` 进行 INT8 PTQ

```bash
python quick_quantize_test.py --demo
```

该命令会运行一个 dummy model 演示各种量化选项的效果。

### 3. 完整校准 + PTQ + 评估

```bash
# 第一步：生成校准数据
# 注意：需要接入真实数据集加载逻辑，当前脚本为框架
python ptq_calibrate_and_eval.py --mode calibrate --dataset visevent --num_sequences 10 --frames_per_seq 30

# 第二步：量化并评估
python ptq_calibrate_and_eval.py --mode ptq_eval --quant_method torchao_int8 --calib_data calib_data.pt
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `quick_quantize_test.py` | 快速量化工具函数 + dummy 演示。可直接集成到 tracker 中。 |
| `ptq_calibrate_and_eval.py` | 完整 PTQ 流程：校准数据生成 → 校准 → 量化 → 评估框架。 |
| `README.md` | 本文件 |

---

## 量化策略建议

对于课程项目的研究探索，建议按以下顺序实验：

1. **FP16 / BF16** — 一键转换，体积减半，几乎无精度损失。
2. **Dynamic INT8 (Weight-Only)** — 一键完成，无需校准数据，主要压缩模型体积。
3. **torchao INT8 (W8A8)** — 需要校准数据，尝试全模型 INT8，评估精度损失。
4. **分层混合精度** — Event 分支和 Prompt 块保留 FP16，Transformer 主干用 INT8。
5. **QAT (可选)** — 如 PTQ 掉点明显，可在 VisEvent 上微调 1-3 epoch 恢复精度。

---

## 注意事项

- **GPU 加速**：PyTorch Eager Mode 的 INT8 在 GPU 上不一定比 FP16 快。真正的 INT8 加速需要 TensorRT、ONNX Runtime 或专门的 INT8 kernel。
- **校准数据**：必须使用与测试时**相同的预处理**（`PreprocessorMM`），否则校准统计量无效。
- **算子支持**：ViPT 包含 `CEBlock` 和 `Prompt_block` 等自定义模块，FX Graph Mode 可能无法完全 tracing。推荐使用 `torchao` 或 `torch.quantization.quantize_dynamic`。
- **事件模态**：Event 数据经过 `patch_embed_prompt` 后分布与 RGB 不同，建议单独研究 Event 分支对量化的敏感度。

---

## 研究问题

1. **事件模态分支是否对量化更敏感？**（由于其稀疏性特点）
2. **Prompt 模块（0.84M 参数）量化后是否仍能有效融合 RGB 与 Event？**
3. **在线模板更新机制在量化模型中是否仍稳定？**（SSIM 阈值是否需调整）

这些问题的实验结论可直接写入课程设计报告。
