# autoresearch — Claude on a 3090

A fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch) where [Claude](https://claude.ai/code) is the autonomous research agent, running experiments on a single RTX 3090 (24GB VRAM).

The setup: Claude reads the codebase, forms hypotheses about what architectural or training changes might lower validation loss, edits `train.py`, runs a 5-minute training run, evaluates, and decides whether to keep or discard. No human in the loop during experimentation — just an AI agent iterating on a real LLM training setup.

## Results

Starting from Karpathy's baseline tuned for RTX 3090, Claude ran 20 experiments across architecture, optimization, and training dynamics. 5 changes survived:

| # | Commit | val_bpb | Delta | Status | What changed |
|---|--------|---------|-------|--------|-------------|
| 0 | `0635a47` | 1.1136 | — | baseline | RTX 3090 tuned defaults |
| 1 | `d68118d` | 1.1108 | -0.0028 | keep | depth=8, batch=48, use more VRAM |
| 2 | `43ef022` | 1.1081 | -0.0027 | keep | SwiGLU MLP replaces ReLU-squared |
| 3 | `acd70ee` | 1.1043 | -0.0038 | keep | shorter sliding window, seq//4 |
| 4 | `8c8c707` | 1.1033 | -0.0010 | keep | post-norm hybrid, Gemma 2 style |
| 5 | `9c4ee17` | 1.1006 | -0.0026 | keep | remove x0 residual shortcut |

**Total improvement: -0.013 val_bpb (1.16%)** over the 3090 baseline.

### What didn't work — 13 discarded experiments

These are arguably more interesting than what did work:

- **Going deeper (depth=10)** made things worse. Value embeddings scale with depth and consume ~38% of total params, so adding layers doesn't pay off within the 5-minute budget — too few steps to converge.
- **Wider models (aspect=64)** also worse — similar VRAM pressure, fewer steps.
- **Hyperparameter tweaks** — higher matrix LR, shorter warmdown, added warmup, different window patterns — were all neutral. The defaults are well-tuned.
- **Removing value embeddings** hurt at depth=8, and deeper models without VE still couldn't overcome the step-count disadvantage. The VE overhead is real, but removing them naively doesn't help yet.
- **Cosine decay schedule** and **increased softcap** — no improvement.
- **Smaller head_dim (64, giving 6 heads)** — slightly worse.

Full experiment log in `results.tsv`.

## What's in the diff

All changes are in `train.py` (the only file the agent edits). Compared to upstream:

- **SwiGLU MLP**: Gated activation with 8/3 expansion ratio, replacing ReLU-squared
- **Post-norm hybrid**: RMSNorm after both attention and MLP outputs (Gemma 2 style), on top of pre-norm
- **Shorter sliding window**: Short layers use `seq_len // 4` instead of `seq_len // 2`
- **Simplified residuals**: Removed `x0` shortcut — each layer is just `lambda * x` instead of `lambda * x + lambda0 * x0`
- **Compute allocation**: depth=8, batch=48, 13.5GB VRAM

## Setup

Same as upstream — requires a single NVIDIA GPU, Python 3.10+, and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run prepare.py   # one-time data prep
uv run train.py     # 5-minute training run
```

## How it works

See the [upstream README](https://github.com/karpathy/autoresearch) for the full explanation. The short version: `program.md` tells Claude how to run the experiment loop, Claude modifies `train.py`, trains for 5 minutes, evaluates, keeps or discards, repeats.

## Status

Active. More experiments coming — the value embedding bottleneck is the most interesting open question. Current hypothesis: VE removal needs to be paired with a compensating change (more depth or a different residual mixing strategy) to actually pay off.

## Credit

- [Andrej Karpathy](https://github.com/karpathy) for autoresearch and the original training setup
- [Claude](https://claude.ai/code) (Anthropic) as the autonomous research agent
- Hardware: RTX 3090 24GB, WSL2 Ubuntu 22.04
