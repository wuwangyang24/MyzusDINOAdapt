# train_synthesis_classifier.py — Usage Guide

Train a Transformer-based classifier to predict the **synthesis program** of a compound from its DINO image embeddings.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Workflow overview](#workflow-overview)
3. [Required inputs](#required-inputs)
4. [Model architecture](#model-architecture)
5. [Arguments reference](#arguments-reference)
6. [Examples](#examples)
7. [Output files](#output-files)
8. [Loading the trained model for inference](#loading-the-trained-model-for-inference)
9. [Tips & troubleshooting](#tips--troubleshooting)

---

## Prerequisites

1. Run `encode_embeddings.py` first to produce the `.pt` embeddings file.
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run from the repo root (or anywhere — the script auto-adds the repo root to `sys.path`):
   ```bash
   python Experiments/train_synthesis_classifier.py --help
   ```

---

## Workflow overview

```
embeddings.pt                    compound_metadata.csv
     │                                    │
     │   { compound_id:                   │   compound | synthesis_program
     │     { plate_id:                    │   ─────────────────────────────
     │       treated: (N, D)  ──┐         │   1        | ProgramA
     │       control: (D,)   ──┘│         │   2        | ProgramB
     │     }                    │         │   ...
     │   }                      │         │
     └──────────────────────────┴─────────┘
                                │
                    CompoundSynthesisDataset
                    (one sample = one compound)
                                │
              [optional] subtract per-plate control mean
                                │
                bag of treated latents  (M, D)   ← M varies per compound
                                │
                  collate_fn: pad → (B, max_M, D)
                              mask → (B, max_M)
                                │
                ┌───────────────────────────────┐
                │  SynthesisProgramClassifier   │
                │                               │
                │  Linear(D → d_model)          │
                │  Prepend [CLS] token          │
                │  Transformer Encoder          │  ← no positional encoding
                │    (set-based, order-free)    │
                │  CLS output → FC head         │
                └───────────────────────────────┘
                                │
                         logits (num_classes)
```

**Key design points:**
- Each compound is represented as an **unordered bag** of treated image embeddings across all its plates.
- Variable bag sizes are handled via padding + attention masking — compounds with 5 images and 500 images coexist in the same batch.
- **No positional encoding** is used, which is intentional: individual well images have no meaningful ordering.
- The `[CLS]` token aggregates information from the entire bag through self-attention.

---

## Required inputs

### 1. Embeddings file (`--embeddings`)

Output of `encode_embeddings.py`. A `.pt` file with this structure:

```python
{
    compound_id (int): {
        plate_id (str): {
            "treated": torch.Tensor,   # shape (N, D)
            "control": torch.Tensor,   # shape (D,)  — averaged controls
        },
        ...
    },
    ...
}
```

### 2. Metadata file (`--metadata`)

A CSV or Excel (`.xlsx`) file with at least two columns:

| compound | synthesis_program |
|----------|------------------|
| 1        | ProgramA         |
| 2        | ProgramB         |
| 3        | ProgramA         |
| ...      | ...              |

- Column names are configurable via `--compound_col` and `--label_col`.
- Compound IDs must match the integer keys in the embeddings file.
- Rows with missing values are automatically dropped.

---

## Model architecture

```
Input bag  (M, D)
    │
    ├── Linear projection       (D → d_model)
    │
    ├── Prepend learnable [CLS] token  →  (M+1, d_model)
    │
    ├── Transformer Encoder  ×  num_layers
    │     • Multi-Head Self-Attention  (nhead heads)
    │     • Feed-Forward Network       (dim_feedforward hidden units)
    │     • Pre-norm (LayerNorm before each sub-layer)
    │     • Padding mask applied  →  padded positions ignored
    │     • No positional encoding  →  purely set-based
    │
    ├── CLS token output  (d_model,)
    │
    └── Classification head
          LayerNorm → Linear(d_model → d_model//2) → GELU → Dropout → Linear(→ num_classes)
```

Default hyper-parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `d_model` | 256 | Transformer model dimension |
| `nhead` | 8 | Number of attention heads |
| `num_layers` | 4 | Number of Transformer encoder layers |
| `dim_feedforward` | 1024 | FFN hidden dimension |
| `dropout` | 0.1 | Dropout rate |

---

## Arguments reference

### Required

| Argument | Description |
|----------|-------------|
| `--embeddings PATH` | Path to the `.pt` embeddings file |
| `--metadata PATH` | Path to the compound metadata CSV or Excel file |

### Data

| Argument | Default | Description |
|----------|---------|-------------|
| `--compound_col` | `compound` | Column name for compound IDs in the metadata |
| `--label_col` | `synthesis_program` | Column name for synthesis program labels |
| `--subtract_control` | off | If set, subtract the per-plate averaged control embedding from each treated embedding before training |
| `--val_split` | `0.2` | Fraction of compounds held out for validation |

### Model

| Argument | Default | Description |
|----------|---------|-------------|
| `--d_model` | `256` | Transformer hidden dimension |
| `--nhead` | `8` | Number of self-attention heads (`d_model` must be divisible by `nhead`) |
| `--num_layers` | `4` | Number of Transformer encoder layers |
| `--dim_feedforward` | `1024` | FFN hidden dimension |
| `--dropout` | `0.1` | Dropout applied in the Transformer and classification head |

### Training

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | `100` | Maximum number of training epochs |
| `--batch_size` | `16` | Number of compounds per batch |
| `--lr` | `3e-4` | AdamW learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--patience` | `20` | Early stopping: epochs without validation improvement before stopping |
| `--label_smoothing` | `0.1` | Label smoothing for `CrossEntropyLoss` |

### Misc

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `Experiments/runs/classifier` | Directory for all saved outputs |
| `--device` | auto | Torch device: `cuda`, `cuda:1`, `cpu`, etc. |
| `--seed` | `42` | Random seed for reproducibility |
| `--num_workers` | `0` | DataLoader worker processes |
| `--save_predictions` | off | If set, save `predictions.csv` with compound IDs, true labels, predicted labels, and per-class probabilities |

---

## Examples

### 1 — Minimal run

```bash
python Experiments/train_synthesis_classifier.py \
    --embeddings Experiments/embeddings.pt \
    --metadata   data/compound_metadata.csv
```

### 2 — With control subtraction

```bash
python Experiments/train_synthesis_classifier.py \
    --embeddings       Experiments/embeddings.pt \
    --metadata         data/compound_metadata.csv \
    --subtract_control
```

### 3 — Custom metadata column names

```bash
python Experiments/train_synthesis_classifier.py \
    --embeddings    Experiments/embeddings.pt \
    --metadata      data/compound_metadata.csv \
    --compound_col  compound_id \
    --label_col     program
```

### 4 — Custom transformer and training hyper-parameters

```bash
python Experiments/train_synthesis_classifier.py \
    --embeddings     Experiments/embeddings.pt \
    --metadata       data/compound_metadata.csv \
    --d_model        512 \
    --nhead          8 \
    --num_layers     6 \
    --dim_feedforward 2048 \
    --dropout        0.15 \
    --epochs         200 \
    --lr             1e-4 \
    --batch_size     32 \
    --patience       30 \
    --output_dir     Experiments/runs/exp1
```

### 5 — Excel metadata, specific GPU

```bash
python Experiments/train_synthesis_classifier.py \
    --embeddings Experiments/embeddings.pt \
    --metadata   data/compound_metadata.xlsx \
    --device     cuda:1 \
    --output_dir Experiments/runs/exp2
```

---

## Output files

All outputs are saved to `--output_dir` (default: `Experiments/runs/classifier/`):

| File | Description |
|------|-------------|
| `best_model.pt` | State-dict of the model with the highest validation accuracy |
| `label_encoder.json` | Maps synthesis programs ↔ integer indices |
| `training_log.csv` | Per-epoch: train/val loss, accuracy, and weighted F1 |
| `predictions.csv` | *(optional, requires `--save_predictions`)* Compound-level predictions with ground truth and class probabilities |

### `label_encoder.json` format

```json
{
  "classes": ["ProgramA", "ProgramB", "ProgramC"],
  "str2idx": {
    "ProgramA": 0,
    "ProgramB": 1,
    "ProgramC": 2
  }
}
```

### `training_log.csv` columns

`epoch, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1`

### `predictions.csv` columns *(requires `--save_predictions`)*

| Column | Description |
|--------|-------------|
| `compound_id` | Integer compound identifier |
| `true_label` | Ground-truth synthesis program name |
| `predicted_label` | Model-predicted synthesis program name |
| `correct` | `True` / `False` — whether prediction matches ground truth |
| `prob_<ProgramA>` | Softmax probability for each class (one column per class) |

Example:

| compound_id | true_label | predicted_label | correct | prob_ProgramA | prob_ProgramB |
|-------------|------------|-----------------|---------|---------------|---------------|
| 42          | ProgramA   | ProgramA        | True    | 0.87          | 0.13          |
| 7           | ProgramB   | ProgramA        | False   | 0.61          | 0.39          |

---

## Loading the trained model for inference

```python
import json
import torch
from Experiments.train_synthesis_classifier import SynthesisProgramClassifier

# Load label encoder
with open("Experiments/runs/classifier/label_encoder.json") as f:
    le = json.load(f)
classes   = le["classes"]       # e.g. ["ProgramA", "ProgramB", ...]
num_classes = len(classes)

# Rebuild model (use the same hyper-parameters as training)
model = SynthesisProgramClassifier(
    input_dim=768,           # D from the backbone (e.g. dino_vitb16 → 768)
    num_classes=num_classes,
    d_model=256,
    nhead=8,
    num_layers=4,
)
model.load_state_dict(
    torch.load("Experiments/runs/classifier/best_model.pt", map_location="cpu")
)
model.eval()

# Inference for a single compound
# treated_latents: (M, D) tensor of treated image embeddings
with torch.no_grad():
    logits = model(treated_latents.unsqueeze(0))   # (1, M, D) → (1, num_classes)
    pred_idx   = logits.argmax(dim=1).item()
    pred_class = classes[pred_idx]
    print(f"Predicted synthesis program: {pred_class}")
```

---

## Tips & troubleshooting

| Problem | Solution |
|---------|----------|
| `ValueError: Metadata is missing column(s)` | Check `--compound_col` and `--label_col` match your CSV headers exactly (case-sensitive) |
| `RuntimeError: Dataset is empty` | Verify compound IDs in the embeddings `.pt` file are integers matching the metadata `compound` column |
| CUDA out of memory | Reduce `--batch_size`; compounds with many treated images consume more memory |
| `d_model` not divisible by `nhead` | `d_model` must be divisible by `nhead` (e.g. d_model=256, nhead=8 ✓) |
| Training loss not decreasing | Try lowering `--lr`, increasing `--num_layers`, or enabling `--subtract_control` |
| Only one class predicted | Dataset may be class-imbalanced — consider increasing `--label_smoothing` or oversampling |
| Want to disable early stopping | Set `--patience` to a value larger than `--epochs` |
