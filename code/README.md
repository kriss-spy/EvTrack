# Code

Tracker implementations and evaluation scripts.

## Structure

```
code/
├── SDSTrack/   # SDSTrack reproduction (upstream submodule + eval scripts)
└── ViPT/       # Pointer to the ViPT fork (kriss-spy/ViPT vipt-improvement)
```

## ViPT

ViPT reproduction lives in a separate fork:

> **https://github.com/kriss-spy/ViPT** — branch [`vipt-improvement`](https://github.com/kriss-spy/ViPT/tree/vipt-improvement)

See [`ViPT/README.md`](./ViPT/README.md) for details.

## SDSTrack

SDSTrack reproduction is self-contained in [`SDSTrack/`](./SDSTrack/).
The upstream code is included as a git submodule.

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/kriss-spy/EvTrack
# Or if already cloned:
git submodule update --init code/SDSTrack/upstream
```

Evaluation results and metrics live in [`experiments/sdstrack/`](../experiments/sdstrack/).
Metric computation scripts are in [`scripts/`](../scripts/).
