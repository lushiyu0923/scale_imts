# ScaleIMTS

ScaleIMTS is an irregular multivariate time-series model with two coordinated
branches:

- **Foundation branch**: a GPT-2/BERT-style backbone models variable-wise tokens.
- **Irregular branch**: patch-level temporal encoding and dynamic cross-variable
graph propagation model sparse observations.
- **Scale-aware fusion**: routing, feature reduction and FiLM-style fusion join
the two branches for forecasting or classification.

This directory is organized for publication and GitHub release. **Datasets,
pretrained weights and checkpoints are intentionally external** and are never
committed here.

## Layout

```text
scaleimts/
├── main.py                       # forecasting entry point
├── main_classification.py        # classification entry point
├── preprocessing/                # external-data validation helpers
├── data/                         # loaders, collators, normalization, perturbation
├── models/core/                  # ScaleIMTS model and its submodules
├── training/                     # forecasting/classification loops and config
├── utils/                        # losses, metrics, logging and checkpoints
├── tests/                        # lightweight structural tests
├── requirements.txt
└── .gitignore
```

## Install

```bash
cd scaleimts
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
# Or install the package in editable mode:
# pip install -e .
```

## Data location

Keep data outside this folder. The loaders accept an explicit path, so the
same code works on a local workstation, cluster, or GitHub checkout.

Forecasting examples:

```bash
python scaleimts/preprocessing/check_dataset.py \
  --task forecasting --dataset USHCN --data-root E:\\datasets\\USHCN
python scaleimts/main.py \
  --dataset USHCN --data-root E:\\datasets\\USHCN \
  --foundation-pretrained-root E:\\models\\PLMs \
  --epochs 10 --device cuda
```

The forecasting data root must contain the files expected by the selected
loader, for example `small_chunked_sporadic.csv` for USHCN,
`set-a.tar.gz`/`set-b.tar.gz`/`set-c.tar.gz` for P12,
`ConfLongDemo_JSI.txt` for HumanActivity, or `complete_tensor.csv` for MIMIC-III.

Classification examples:

```bash
python scaleimts/preprocessing/check_dataset.py \
  --task classification --dataset P12 --data-root E:\\datasets\\classification
python scaleimts/main_classification.py \
  --dataset P12 --data-root E:\\datasets\\classification \
  --foundation-pretrained-root E:\\models\\PLMs \
  --split 1 --epochs 20 --device cuda
```

For classification, the data root can contain `P12/`, `P19/`, `PAM/` and
`MIMIC_III/` subdirectories. `SCALEIMTS_DATA_ROOT` can be used instead of
`--data-root` when calling the lower-level loader API.

## Pretrained backbones

`--foundation-pretrained-root` should contain `gpt2/` and
`bert-base-uncased/` directories with Hugging Face configuration and weights.
If it is omitted, ScaleIMTS initializes a compatible random GPT-2/BERT backbone,
which is useful for smoke tests but not a pretrained comparison. For local
pretrained weights, set `--hidden-dim 768` unless the checkpoint uses another
hidden size.

The loader first tries the original `*_wope` classes when they are available;
otherwise it uses the public Transformers GPT-2/BERT implementations. This
removes the previous dependency on a modified global `transformers` install.

## Evaluation and checkpoints

Each run writes `best_model.pt` below `--save-dir`, separated by task, dataset,
fusion type and classification split. Forecasting reports MAE/MSE;
classification reports accuracy, AUROC, AUPRC, precision, recall and F1.

## Tests

```bash
python -m pytest tests
```

The test suite is intentionally data-independent. Downloaded data and model
weights should remain outside the repository and be supplied through paths.
