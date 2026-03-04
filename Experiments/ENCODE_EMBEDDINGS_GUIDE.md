# encode_embeddings.py — Usage Guide

Encodes microscopy images using a pretrained DINO backbone (optionally with LoRA or DoRA adaptation) and saves the resulting embeddings to a single `.pt` file.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Input — Metadata format](#input--metadata-format)
3. [Output — Embedding file format](#output--embedding-file-format)
4. [Arguments reference](#arguments-reference)
5. [Examples](#examples)
6. [Loading and using the output file](#loading-and-using-the-output-file)
7. [Tips & troubleshooting](#tips--troubleshooting)

---

## Prerequisites

Install the project dependencies from the repo root before running the script:

```bash
pip install -r requirements.txt
```

The script must be run from **the repo root** (or anywhere — it automatically adds the repo root to `sys.path`):

```bash
# from repo root
python Experiments/encode_embeddings.py --help
```

---

## Input — Metadata format

The metadata file must be a **JSON file** containing either:

- A **list** of compound dicts, or
- A **dict** with a `"compounds"` key that holds the list.

### Format A — bare list (recommended)

```json
[
    {
        "Compound": 1,
        "94000": {
            "treated": [
                "94000/well_2_1/treated/sample_1.png",
                "94000/well_2_1/treated/sample_2.png"
            ],
            "control": [
                "94000/well_1_3/control/sample_1.png",
                "94000/well_1_3/control/sample_2.png"
            ]
        },
        "131000": {
            "treated": ["131000/well_3_6/treated/sample_1.png"],
            "control": ["131000/well_1_2/control/sample_1.png"]
        }
    },
    {
        "Compound": 2,
        "94000": { "treated": [...], "control": [...] }
    }
]
```

### Format B — wrapped dict

```json
{
    "compounds": [ ... same list as above ... ]
}
```

**Key rules:**
- Every entry must have a `"Compound"` key (string ID).
- All other top-level keys are treated as **plate identifiers** (strings).
- Image paths are **relative to `--root_dir`**.
- Both `"treated"` and `"control"` lists are optional per plate, but a warning is printed if either is missing.

---

## Output — Embedding file format

The output is a PyTorch `.pt` file containing a nested Python dict:

```
{
    <compound_id: str>: {
        <plate_id: str>: {
            "treated": torch.Tensor,   # shape (N, D) — one row per treated image
            "control": torch.Tensor,   # shape (D,)   — mean of all control embeddings
        },
        ...
    },
    ...
}
```

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `treated` | `torch.FloatTensor` | `(N, D)` | One embedding per treated image |
| `control` | `torch.FloatTensor` | `(D,)` | Mean embedding across all control images on that plate |

`D` is the feature dimension of the backbone (see table below).

| Backbone | D |
|----------|---|
| `dino_vits8` / `dino_vits16` | 384 |
| `dino_vitb8` / `dino_vitb16` | 768 |
| `dino_vitl14` | 1024 |
| `dino_vitg14` | 1536 |

---

## Arguments reference

### Required

| Argument | Description |
|----------|-------------|
| `--metadata PATH` | Path to the metadata JSON file |
| `--root_dir PATH` | Root directory that image paths in the metadata are relative to |

### Model selection

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_type` | `dino` | Encoder to use: `dino`, `dino_lora`, or `dino_dora` |
| `--backbone` | `dino_vitb16` | DINO backbone variant (see table above) |
| `--weights_path PATH` | `None` | Path to a fine-tuned checkpoint (`.pt`/`.pth`). Used with `dino_lora` / `dino_dora` |
| `--hub_source` | `github` | Where to load the backbone from: `github` or `local` |
| `--hub_source_dir PATH` | `None` | Local DINO hub directory (required when `--hub_source local`) |

### LoRA hyper-parameters *(used only with `--model_type dino_lora`)*

| Argument | Default | Description |
|----------|---------|-------------|
| `--lora_r` | `8` | Rank of the low-rank matrices |
| `--lora_alpha` | `16.0` | Scaling factor |
| `--lora_dropout` | `0.1` | Dropout rate |

> These must match the values used during training so that the checkpoint loads correctly.

### DoRA hyper-parameters *(used only with `--model_type dino_dora`)*

| Argument | Default | Description |
|----------|---------|-------------|
| `--dora_r` | `8` | Rank of the low-rank matrices |
| `--dora_alpha` | `16.0` | Scaling factor |
| `--dora_dropout` | `0.1` | Dropout rate |

### Misc

| Argument | Default | Description |
|----------|---------|-------------|
| `--output PATH` | `embeddings.pt` | Output `.pt` file path |
| `--batch_size` | `64` | Images per forward pass — reduce if you hit OOM |
| `--device` | auto | PyTorch device: `cuda`, `cuda:1`, `cpu`, etc. |

---

## Examples

### 1 — Plain pretrained DINO (no adaptation)

```bash
python Experiments/encode_embeddings.py \
    --metadata  data/metadata.json \
    --root_dir  data/images \
    --output    Experiments/dino_embeddings.pt \
    --backbone  dino_vitb16 \
    --batch_size 64
```

### 2 — DINO + LoRA with a fine-tuned checkpoint

```bash
python Experiments/encode_embeddings.py \
    --metadata     data/metadata.json \
    --root_dir     data/images \
    --output       Experiments/lora_embeddings.pt \
    --model_type   dino_lora \
    --backbone     dino_vitb16 \
    --weights_path runs/lora_exp1/best_model.pt \
    --lora_r       8 \
    --lora_alpha   16.0
```

### 3 — DINO + DoRA with a fine-tuned checkpoint

```bash
python Experiments/encode_embeddings.py \
    --metadata     data/metadata.json \
    --root_dir     data/images \
    --output       Experiments/dora_embeddings.pt \
    --model_type   dino_dora \
    --backbone     dino_vitb16 \
    --weights_path runs/dora_exp1/best_model.pt \
    --dora_r       8 \
    --dora_alpha   16.0
```

### 4 — Using a locally cached DINO hub (no internet needed)

```bash
python Experiments/encode_embeddings.py \
    --metadata       data/metadata.json \
    --root_dir       data/images \
    --output         Experiments/dino_embeddings.pt \
    --hub_source     local \
    --hub_source_dir /path/to/local/dino_hub \
    --backbone       dino_vitb16
```

### 5 — CPU-only, small batch

```bash
python Experiments/encode_embeddings.py \
    --metadata   data/metadata.json \
    --root_dir   data/images \
    --output     Experiments/embeddings.pt \
    --device     cpu \
    --batch_size 16
```

---

## Loading and using the output file

```python
import torch

embeddings = torch.load("Experiments/embeddings.pt")

# Iterate over compounds
for compound_id, plates in embeddings.items():
    print(f"Compound {compound_id}")
    for plate_id, data in plates.items():
        treated = data["treated"]   # (N, D)
        control = data["control"]   # (D,)
        print(f"  Plate {plate_id}: "
              f"{treated.shape[0]} treated images, "
              f"control embedding dim={control.shape[0]}")
```

### Typical downstream use — subtract control from treated

```python
for compound_id, plates in embeddings.items():
    for plate_id, data in plates.items():
        treated = data["treated"]        # (N, D)
        control = data["control"]        # (D,)
        adjusted = treated - control     # broadcast subtraction → (N, D)
        plate_mean = adjusted.mean(0)    # (D,) — plate-level representation
```

---

## Tips & troubleshooting

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce `--batch_size` (e.g. `--batch_size 16`) |
| `FileNotFoundError` for an image | Check that `--root_dir` is correct and that paths in the metadata are relative to it |
| Checkpoint load warnings about missing/unexpected keys | Ensure `--lora_r`, `--lora_alpha`, `--backbone` match the training config |
| `ModuleNotFoundError: src` | Run the script from the repo root, or ensure `src/` is on `PYTHONPATH` |
| Slow download of DINO weights | Pass `--hub_source local --hub_source_dir /cached/dino` to skip the download |
| Want to inspect feature dimensions | `print(embeddings[compound_id][plate_id]["treated"].shape)` |
