# Documentation

This directory holds the project's written documentation: proposal, course
requirements, dataset setup, background reading notes, and research
exploration notes.

Tracker-specific documentation lives next to the code and experiments, and is
linked from the index below.

## Project & Course

| Document | Description |
|----------|-------------|
| [report.md](report.md) | Final project report (forthcoming, Issue #27) |
| [project-proposal.md](project-proposal.md) | 开题报告 — project proposal, background, objectives, schedule (Chinese) |
| [course-project-guide.md](course-project-guide.md) | Course design workflow, grading rubric, and submission requirements (Chinese) |
| [《模式识别》课程设计指导书（2026）修订版1.pdf](《模式识别》课程设计指导书（2026）修订版1.pdf) | Official course design handbook (PDF) |

## Datasets

| Document | Description |
|----------|-------------|
| [dataset-setup.md](dataset-setup.md) | Download and preparation guide for VisEvent and COESOT |

## Background & Survey Notes

| Document | Description |
|----------|-------------|
| [Event-Based Vision.md](Event-Based Vision.md) | Reading notes on the Gallego et al. event-camera survey: operating principle, event representations, and processing paradigms (Chinese) |

## Research Exploration

| Document | Description |
|----------|-------------|
| [vipt-quantization-research.md](vipt-quantization-research.md) | ViPT quantization study plan (Issue #9): FP16/INT8 PTQ/QAT roadmap and experiment design (Chinese) |

## Tracker & Experiment Documentation

These live alongside the code and experiment artifacts, but are part of the
project's documentation surface.

| Document | Location | Description |
|----------|----------|-------------|
| Code overview | [code/README.md](../code/README.md) | Tracker implementations and eval scripts layout |
| ViPT fork | [code/ViPT/README.md](../code/ViPT/README.md) | ViPT reproduction fork + online-template improvement |
| SDSTrack | [code/SDSTrack/README.md](../code/SDSTrack/README.md) | SDSTrack reproduction setup and quick start |
| SDSTrack experiment | [experiments/sdstrack/README.md](../experiments/sdstrack/README.md) | VisEvent reproduction results, metrics, and archive |
| Reproduction log | [experiments/sdstrack/REPRODUCTION_LOG.md](../experiments/sdstrack/REPRODUCTION_LOG.md) | Step-by-step SDSTrack reproduction timeline |
| Environment snapshot | [experiments/sdstrack/ENVIRONMENT.md](../experiments/sdstrack/ENVIRONMENT.md) | Software environment used for SDSTrack eval |

## Notes

- Literature that is comfortably managed in the Zotero group library does not
  need to be duplicated here.
- Course-facing narrative content in this directory is in Chinese; the
  root [README](../README.md) is in English.
