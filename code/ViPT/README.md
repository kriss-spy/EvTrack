# ViPT

ViPT (Visual Prompt Multi-Modal Tracking) reproduction lives in a fork of the
[upstream ViPT](https://github.com/jiawen-zhu/ViPT) repository:

> **https://github.com/kriss-spy/ViPT** — branch [`vipt-improvement`](https://github.com/kriss-spy/ViPT/tree/vipt-improvement)

The fork contains the original ViPT code plus our online-template improvement
(additional online RGB+event template pair with SSIM-gated dynamic updates,
no extra training required).

## Clone

```bash
git clone -b vipt-improvement https://github.com/kriss-spy/ViPT
```

## Key changes vs upstream

See the [commits on `vipt-improvement`](https://github.com/kriss-spy/ViPT/commits/vipt-improvement):

- `lib/test/tracker/vipt_online_template.py` — new tracker with online template update
- `lib/models/vipt/vit_ce_prompt.py` — backbone support for online template tokens
- `lib/models/vipt/ostrack_prompt.py` — forward pass threading for online template
- `lib/models/layers/attn_blocks.py` — CE block handles online template tokens
- `RGBE_workspace/test_rgbe_mgpus.py` — switched to `vipt_online_template` tracker
- `--seq_home` / `--save_dir` CLI args for eval scripts (no more hardcoded paths)

## Usage

```bash
cd ViPT
bash eval_rgbe.sh --seq_home /path/to/VisEvent/test --save_dir ./RGBE_workspace/results
```

Results and GIFs comparing original vs improved tracking are in the EvTrack repo
(`gif/` directory) and the root `README.md`.
