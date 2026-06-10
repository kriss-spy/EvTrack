`ostrack_prompt.py`和 `vit_ce_prompt`放在 `vipt/lib/models/vipt`,直接进行覆盖即可，做了很好的后向支持

`vipt_online_template.py`和 `vipt_single_template`放在 `vipt/lib/test/tracker`

在 `RGBE_workspace/test_rgbe_mgpus.py`中进行修改 `from lib.test.tracker`.`vipt_online_template import ViPTTrack`，使用原始的就用 `from vipt`，使用新增的就 `vipt_single_template`或者 `vipt_online_template`，不建议使用single版，效果不好，使用online版可以

关于如何在自己的数据集上跑，可以阅读vipt教程，进行一定的修改即可

---

## FP16 混合精度推理

`vipt_online_template.py` 和 `vipt_single_template.py` 已支持 FP16 混合精度推理。

### 开启方式

在 tracker 参数中传入 `use_fp16=True`：

```python
params.use_fp16 = True  # 默认 False，保持原有 FP32 行为
```

或在 `test_rgbe_mgpus.py` 等测试脚本中通过命令行参数传入：

```python
parser.add_argument('--use_fp16', action='store_true', help='启用 FP16 混合精度推理')
# ...
params.use_fp16 = args.use_fp16
```

### 实现说明

- `use_fp16=False`（默认）：完全保持原有 FP32 行为，后向兼容。
- `use_fp16=True`：
  - 模型权重加载后通过 `.half()` 转为 FP16
  - `hann2d` 输出窗口同步转为 FP16
  - 输入 `template` / `search` / `target_tensor` 在送入网络前自动转为 FP16
  - 在线模板更新（`online_z_tensor`）仍保持 FP16，无需额外转换
  - 其他后处理逻辑（`SSIM`、`save_img`）不受影响，因为比较/保存操作在 CPU 侧进行

### 效果

- **模型体积**：理论上减半（从 4 bytes/float 到 2 bytes/float）
- **显存占用**：权重显存减半，激活显存同步减半
- **推理速度**：在支持 Tensor Core 的 GPU 上（如 RTX 30/40 系列、A100 等）通常有 1.3x~2x 加速
- **精度影响**：对 ViT-based 跟踪器通常几乎无损失（Success/Precision 掉点 < 0.5%）

### 注意事项

1. 需要 GPU 支持 FP16（现代 NVIDIA GPU 均支持）
2. 如果观察到精度明显下降，建议检查 `LayerNorm` 和 `Softmax` 层是否数值不稳定（可尝试用 `torch.autocast` 包裹而非全模型 `.half()`）
3. `vot` 包相关的调试代码不受影响
