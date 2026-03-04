"""
train_synthesis_classifier.py

Train a Transformer-based classifier to predict the synthesis program of a
compound from its DINO embeddings.

DATA FLOW
---------
For each compound:
  1. Collect all treated latent vectors across every plate  →  (M, D)
     (optionally subtract the per-plate averaged control embedding first)
  2. Prepend a learnable [CLS] token                        →  (M+1, D)
  3. Pass through a Transformer Encoder                     →  (M+1, D)
  4. Take [CLS] output                                      →  (D,) → FC → num_classes

Inputs
------
  --embeddings   Experiments/embeddings.pt
                 Output of encode_embeddings.py:
                    { compound_id: { plate_id: {"treated": (N,D), "control": (D,)} } }

  --metadata     CSV / Excel file with at least two columns:
                    "compound"           (int)  — must match compound_id keys in .pt file
                    "synthesis_program"  (str)  — class label

Usage examples
--------------
  # basic
  python Experiments/train_synthesis_classifier.py \
      --embeddings Experiments/embeddings.pt \
      --metadata   data/compound_metadata.csv \
      --output_dir Experiments/runs/classifier

  # with control subtraction, custom transformer hyper-parameters
  python Experiments/train_synthesis_classifier.py \
      --embeddings       Experiments/embeddings.pt \
      --metadata         data/compound_metadata.csv \
      --subtract_control \
      --d_model 256 --nhead 8 --num_layers 4 \
      --epochs 100 --lr 3e-4 --batch_size 16

Output
------
  <output_dir>/
      best_model.pt       — state-dict of the best validation-accuracy model
      label_encoder.json  — { "classes": [...], "str2idx": {...} }
      training_log.csv    — per-epoch train/val metrics
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report
)
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class CompoundSynthesisDataset(Dataset):
    """
    One sample = one compound.

    Returns
    -------
    latents : torch.FloatTensor  shape (M, D)
        All treated image embeddings across every plate for the compound.
        If subtract_control=True, each treated embedding has its plate's
        averaged control embedding subtracted before stacking.
    label   : int
        Integer class index of the synthesis program.
    compound_id : int
        Original compound identifier (useful for debugging).
    """

    def __init__(
        self,
        embeddings: Dict,
        compound_col: pd.Series,
        label_col: pd.Series,
        label2idx: Dict[str, int],
        subtract_control: bool = False,
    ):
        self.subtract_control = subtract_control
        self.samples: List[Tuple[torch.Tensor, int, int]] = []

        # Build compound → label map from the DataFrame columns
        comp2label: Dict[int, int] = {}
        for comp, prog in zip(compound_col, label_col):
            comp2label[int(comp)] = label2idx[str(prog)]

        for compound_id, plates in embeddings.items():
            cid = int(compound_id)
            if cid not in comp2label:
                continue

            plate_latents: List[torch.Tensor] = []
            for plate_data in plates.values():
                treated = plate_data.get("treated")   # (N, D) or missing
                if treated is None or treated.numel() == 0:
                    continue

                if subtract_control and "control" in plate_data:
                    control = plate_data["control"]   # (D,)
                    treated = treated - control.unsqueeze(0)

                plate_latents.append(treated.float())

            if not plate_latents:
                continue  # compound has no treated images — skip

            all_latents = torch.cat(plate_latents, dim=0)   # (M, D)
            self.samples.append((all_latents, comp2label[cid], cid))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        latents, label, cid = self.samples[idx]
        return latents, label, cid


def collate_fn(
    batch: List[Tuple[torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pad variable-length bags to the max bag size in the batch.

    Returns
    -------
    padded   : (B, max_M, D)
    mask     : (B, max_M)  — True means 'ignore this position' (padding)
    labels   : (B,)
    cids     : (B,)
    """
    latents_list, labels, cids = zip(*batch)
    max_m = max(t.shape[0] for t in latents_list)
    d = latents_list[0].shape[1]

    padded = torch.zeros(len(batch), max_m, d)
    mask = torch.ones(len(batch), max_m, dtype=torch.bool)  # True = padded

    for i, t in enumerate(latents_list):
        m = t.shape[0]
        padded[i, :m] = t
        mask[i, :m] = False                                  # False = real token

    return (
        padded,
        mask,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(cids,   dtype=torch.long),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Model
# ═══════════════════════════════════════════════════════════════════════════════

class SynthesisProgramClassifier(nn.Module):
    """
    Transformer Encoder + CLS-token classifier for compound synthesis programs.

    Architecture
    ------------
    Input projection  : Linear(D, d_model)
    CLS token         : learnable (1, d_model)
    Transformer Enc.  : num_layers × (MultiHeadAttn + FFN)
    Classification    : LayerNorm → Linear(d_model, num_classes)
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project DINO features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,      # (B, S, D) convention
            norm_first=True,       # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args
        ----
        x            : (B, M, D)   — bag of treated embeddings
        padding_mask : (B, M)      — True where positions are padding

        Returns
        -------
        logits : (B, num_classes)
        """
        B = x.shape[0]

        # Project to model dimension
        x = self.input_proj(x)                    # (B, M, d_model)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)    # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)            # (B, M+1, d_model)

        # Extend the padding mask: CLS token is never masked
        if padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # (B, M+1)

        # Transformer
        x = self.transformer(x, src_key_padding_mask=padding_mask)   # (B, M+1, d_model)

        # Take [CLS] output → classify
        cls_out = x[:, 0]                          # (B, d_model)
        logits = self.classifier(cls_out)           # (B, num_classes)
        return logits


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, float]:
    """Run one train or eval epoch. Pass optimizer=None for eval."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: List[int] = []
    all_labels: List[int] = []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for padded, mask, labels, _ in tqdm(
            loader,
            desc="  train" if is_train else "  val  ",
            leave=False,
        ):
            padded = padded.to(device)
            mask   = mask.to(device)
            labels = labels.to(device)

            logits = model(padded, padding_mask=mask)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.detach().argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    return {
        "loss":     total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1":       f1_score(all_labels, all_preds, average="weighted",
                             zero_division=0),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Label encoding helpers
# ═══════════════════════════════════════════════════════════════════════════════

def build_label_encoder(series: pd.Series) -> Tuple[Dict[str, int], List[str]]:
    classes = sorted(series.astype(str).unique().tolist())
    str2idx = {c: i for i, c in enumerate(classes)}
    return str2idx, classes


def save_label_encoder(classes: List[str], str2idx: Dict[str, int], path: Path) -> None:
    with open(path, "w") as f:
        json.dump({"classes": classes, "str2idx": str2idx}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a Transformer classifier for compound synthesis programs."
    )

    # ---- Data ----
    p.add_argument("--embeddings", required=True,
                   help="Path to the .pt embeddings file from encode_embeddings.py")
    p.add_argument("--metadata", required=True,
                   help="CSV or Excel file with 'compound' and 'synthesis_program' columns")
    p.add_argument("--compound_col", default="compound",
                   help="Name of the compound ID column in metadata. Default: compound")
    p.add_argument("--label_col", default="synthesis_program",
                   help="Name of the synthesis program column in metadata. Default: synthesis_program")
    p.add_argument("--subtract_control", action="store_true",
                   help="Subtract per-plate averaged control embedding from treated embeddings")
    p.add_argument("--val_split", type=float, default=0.2,
                   help="Fraction of compounds used for validation. Default: 0.2")

    # ---- Model ----
    p.add_argument("--d_model",         type=int,   default=256,  help="Transformer model dim. Default: 256")
    p.add_argument("--nhead",           type=int,   default=8,    help="Number of attention heads. Default: 8")
    p.add_argument("--num_layers",      type=int,   default=4,    help="Number of transformer layers. Default: 4")
    p.add_argument("--dim_feedforward", type=int,   default=1024, help="FFN hidden dim. Default: 1024")
    p.add_argument("--dropout",         type=float, default=0.1,  help="Dropout rate. Default: 0.1")

    # ---- Training ----
    p.add_argument("--epochs",      type=int,   default=100,  help="Training epochs. Default: 100")
    p.add_argument("--batch_size",  type=int,   default=16,   help="Batch size (compounds). Default: 16")
    p.add_argument("--lr",          type=float, default=3e-4, help="Learning rate. Default: 3e-4")
    p.add_argument("--weight_decay",type=float, default=1e-4, help="AdamW weight decay. Default: 1e-4")
    p.add_argument("--patience",    type=int,   default=20,
                   help="Early-stopping patience (epochs without val improvement). Default: 20")
    p.add_argument("--label_smoothing", type=float, default=0.1,
                   help="Label smoothing for CrossEntropyLoss. Default: 0.1")

    # ---- Misc ----
    p.add_argument("--output_dir",  default="Experiments/runs/classifier",
                   help="Directory for checkpoints and logs. Default: Experiments/runs/classifier")
    p.add_argument("--device",      default=None,
                   help="Torch device. Auto-detected if not specified.")
    p.add_argument("--seed",        type=int, default=42, help="Random seed. Default: 42")
    p.add_argument("--num_workers", type=int, default=0,  help="DataLoader workers. Default: 0")
    p.add_argument("--save_predictions", action="store_true",
                   help="Save validation predictions + ground truth to predictions.csv")

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(
        args.device if args.device else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device : {device}")

    # ── Output directory ─────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load embeddings ───────────────────────────────────────────────────────
    print(f"Loading embeddings : {args.embeddings}")
    embeddings: Dict = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(embeddings)} compounds found in embeddings file.")

    # ── Load metadata ─────────────────────────────────────────────────────────
    print(f"Loading metadata   : {args.metadata}")
    meta_path = Path(args.metadata)
    if meta_path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)

    required_cols = {args.compound_col, args.label_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Metadata is missing column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[args.compound_col, args.label_col]].dropna()
    print(f"  {len(df)} compound rows after dropping NaN.")

    # ── Label encoding ────────────────────────────────────────────────────────
    str2idx, classes = build_label_encoder(df[args.label_col])
    num_classes = len(classes)
    print(f"  {num_classes} synthesis programs: {classes}")
    save_label_encoder(classes, str2idx, output_dir / "label_encoder.json")

    # ── Dataset ───────────────────────────────────────────────────────────────
    full_dataset = CompoundSynthesisDataset(
        embeddings=embeddings,
        compound_col=df[args.compound_col],
        label_col=df[args.label_col],
        label2idx=str2idx,
        subtract_control=args.subtract_control,
    )
    print(f"  {len(full_dataset)} compounds with valid treated embeddings.")

    if len(full_dataset) == 0:
        raise RuntimeError(
            "Dataset is empty. Check that compound IDs in the embeddings file "
            "match the compound IDs in the metadata."
        )

    # ── Train / val split ─────────────────────────────────────────────────────
    n_val   = max(1, int(len(full_dataset) * args.val_split))
    n_train = len(full_dataset) - n_val
    if n_train < 1:
        raise RuntimeError(
            f"Dataset has only {len(full_dataset)} compound(s) — not enough for a "
            f"train/val split of {args.val_split}. Reduce --val_split or add more data."
        )
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    print(f"  Train: {n_train}  |  Val: {n_val}")

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # ── Infer input_dim from first sample ────────────────────────────────────
    sample_latents, _, _ = full_dataset[0]
    input_dim = sample_latents.shape[-1]
    print(f"\nEmbedding dim (D) : {input_dim}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SynthesisProgramClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params  : {n_params:,}")

    # ── Loss, optimizer, scheduler ───────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = -1.0
    patience_counter = 0
    log_rows: List[Dict] = []

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>8}  {'ValF1':>7}")
    print("─" * 62)

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics   = run_epoch(model, val_loader,   criterion, device)
        scheduler.step()

        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()},
               **{f"val_{k}": v for k, v in val_metrics.items()}}
        log_rows.append(row)

        print(f"{epoch:>6}  {train_metrics['loss']:>10.4f}  "
              f"{train_metrics['accuracy']:>9.4f}  "
              f"{val_metrics['loss']:>8.4f}  "
              f"{val_metrics['accuracy']:>8.4f}  "
              f"{val_metrics['f1']:>7.4f}")

        # ── Save best model ───────────────────────────────────────────────────
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            print(f"         ↑ new best val accuracy: {best_val_acc:.4f}  [saved]")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs "
                      f"(patience={args.patience}).")
                break

    # ── Save training log ─────────────────────────────────────────────────────
    pd.DataFrame(log_rows).to_csv(output_dir / "training_log.csv", index=False)

    # ── Final evaluation with best model ──────────────────────────────────────
    print(f"\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False))
    model.eval()

    all_preds:  List[int] = []
    all_labels: List[int] = []
    all_cids:   List[int] = []
    all_probs:  List[List[float]] = []

    with torch.no_grad():
        for padded, mask, labels, cids in val_loader:
            padded, mask = padded.to(device), mask.to(device)
            logits = model(padded, padding_mask=mask)             # (B, num_classes)
            probs  = torch.softmax(logits, dim=1).cpu().tolist()  # (B, num_classes)
            preds  = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())
            all_cids.extend(cids.tolist())
            all_probs.extend(probs)

    print("\nClassification Report (validation set):")
    print(classification_report(
        all_labels, all_preds,
        target_names=classes,
        zero_division=0,
    ))
    print(f"\nBest val accuracy : {best_val_acc:.4f}")

    # ── Save predictions DataFrame ────────────────────────────────────────────
    if args.save_predictions:
        pred_rows = {
            "compound_id":      all_cids,
            "true_label":       [classes[i] for i in all_labels],
            "predicted_label":  [classes[i] for i in all_preds],
            "correct":          [t == p for t, p in zip(all_labels, all_preds)],
        }
        # Add one probability column per class
        prob_array = np.array(all_probs)   # (N, num_classes)
        for cls_idx, cls_name in enumerate(classes):
            pred_rows[f"prob_{cls_name}"] = prob_array[:, cls_idx].tolist()

        pred_df = pd.DataFrame(pred_rows)
        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"Predictions saved to: {pred_path}")

    print(f"Outputs saved to  : {output_dir}")


if __name__ == "__main__":
    main()
