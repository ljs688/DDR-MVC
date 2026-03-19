# DDR-MVC (Minimal)

This repository is a compact DDR-MVC training project with a clean structure.
It keeps only the essentials:

- Multi-view dataset loading (`.mat`)
- DDR-MVC model training
- Clustering evaluation (ACC/NMI/ARI)
- Checkpoint saving

No experiment orchestration scripts, plotting code, or batch runners are included.

## Project Layout

```
DDR-MVC/
  data/
    MSRCV1.mat
  ddr_mvc/
    __init__.py
    config.py
    data.py
    losses.py
    metrics.py
    model.py
    trainer.py
  checkpoints/
  train.py
  requirements.txt
  README.md
```

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run DDR-MVC training:

```bash
python train.py
```

3. Optional arguments:

```bash
python train.py --epochs 100 --batch-size 128 --dataset-name MSRCV1
```

## Notes

- Default dataset is `data/MSRCV1.mat` (smallest existing dataset in the source project).
- Output checkpoint is saved to `checkpoints/ddr_mvc_<dataset>.pt`.
