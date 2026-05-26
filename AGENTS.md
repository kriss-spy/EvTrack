# AGENTS.md

## Project

**EvTrack**: Event Camera-based Object Tracking. The goal is to reproduce and evaluate [ViPT](https://github.com/jiawen-zhu/ViPT) on event-camera tracking benchmarks.

## Key Datasets

- [VisEvent](https://github.com/wangxiao5791509/VisEvent_SOT_Benchmark)
- [COESOT](https://github.com/Event-AHU/COESOT)

## Project Management

- GitHub Issues and GitHub Projects for task tracking.
- `gh` (GitHub CLI) is the universal interface for agents — prefer it for issues, PRs, and project board operations.

## Current Status

- **Milestone 1** (环境搭建与数据准备, due 2026-05-17): **closed** — baseline environment ready.
- **Milestone 2** (方案设计与中期检查, due 2026-05-24): open — past due.
- **Milestone 3** (实验, due 2026-06-14): open.
- **Milestone 4** (评估与改进, due 2026-06-28): open.
- **Milestone 5** (撰写课程设计报告, due 2026-07-05): open.
- **Milestone 6** (考核答辩, due 2026-07-12): open.
- No code in repo yet. Open issues: #3 (ViPT复现) and #5 (文献调研和综述报告).

## Conventions

- Language: Python (`.gitignore` is Python-centric).
- Root `README.md` is English and technical (project-facing).
- Course-related narrative content lives in `docs/` and uses Chinese; keep that convention there.
- Single reproduction track: ViPT. Code lives under `code/`.

## Grading Context

Evaluation weights: workload 15%, individual effort 10%, innovation 20%, experiment & analysis 35%, report writing 10%, defense 10%. Submission requires printed report, slides, and a video demo (≤90s).
