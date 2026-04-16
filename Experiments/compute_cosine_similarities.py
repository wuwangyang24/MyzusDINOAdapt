"""
compute_cosine_similarities.py

Compute intra- and inter-compound cosine similarities from pre-encoded
embeddings (.pt) filtered by a compound list CSV — mirroring the diagnostic
metrics logged during training (see src/training/trainer.py).

Metrics computed (on raw mean embeddings AND on delta vectors):
  - intra_compound_cos_sim  : per-compound cos sim between mean treated
                              embeddings of two plates (same compound).
  - inter_compound_cos_sim  : per-compound mean cos sim of compound average
                              vs all other compounds' averages.
  - intra_delta_cos_sim     : per-compound cos sim between delta vectors
                              (mean_treated − control) of two plates.
  - inter_delta_cos_sim     : per-compound mean cos sim of mean delta
                              vs all other compounds' mean deltas.

Usage:
    python Experiments/compute_cosine_similarities.py \
        --embeddings  /path/to/embeddings.pt \
        --compound_csv Experiments/efficacy_500ppm.csv \
        --output      /path/to/cosine_similarities.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_compound_list(csv_path: str) -> List[str]:
    """Load compound IDs from a CSV with a 'Compound No' column."""
    df = pd.read_csv(csv_path)
    return [str(c) for c in df["Compound No"].tolist()]


def _get_plate_pairs(
    embeddings: Dict,
    compound_ids: List[str],
) -> List[Tuple[str, str, str]]:
    """
    Return (compound_id, plate1, plate2) for compounds that have
    exactly two plates in the embeddings dict.
    """
    pairs = []
    for cid in compound_ids:
        if cid not in embeddings:
            continue
        plates = sorted(embeddings[cid].keys())
        if len(plates) >= 2:
            pairs.append((cid, plates[0], plates[1]))
    return pairs


def compute_similarities(
    embeddings_path: str,
    compound_csv_path: str,
) -> Dict:
    """
    Compute intra/inter cosine similarities on raw embeddings
    and on delta (treated − control) vectors.

    Returns a dict with scalar summaries and per-compound lists.
    """
    embeddings = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    compound_ids = load_compound_list(compound_csv_path)

    pairs = _get_plate_pairs(embeddings, compound_ids)
    if not pairs:
        raise ValueError(
            "No compounds with two plates found in the embeddings. "
            "Check that the compound CSV matches the embeddings file."
        )

    # ── Per-compound: cosine sim between mean treated embeddings of two plates ──
    intra_cos_sims: List[float] = []
    plate_means: List[torch.Tensor] = []

    # ── Per-compound delta vectors ──
    deltas_p1_list: List[torch.Tensor] = []
    deltas_p2_list: List[torch.Tensor] = []

    compound_labels: List[str] = []

    for cid, p1, p2 in tqdm(pairs, desc="Computing per-compound similarities"):
        treated_p1 = embeddings[cid][p1]["treated"].float()  # (N1, D)
        treated_p2 = embeddings[cid][p2]["treated"].float()  # (N2, D)
        control_p1 = embeddings[cid][p1]["control"].float()  # (D,)
        control_p2 = embeddings[cid][p2]["control"].float()  # (D,)

        mean_p1 = treated_p1.mean(dim=0)  # (D,)
        mean_p2 = treated_p2.mean(dim=0)  # (D,)

        # Intra raw cos sim
        cos = F.cosine_similarity(
            mean_p1.unsqueeze(0), mean_p2.unsqueeze(0)
        ).item()
        intra_cos_sims.append(cos)

        # Compound average (used for inter)
        plate_means.append((mean_p1 + mean_p2) / 2.0)

        # Delta vectors: mean_treated − control
        delta_p1 = mean_p1 - control_p1
        delta_p2 = mean_p2 - control_p2
        deltas_p1_list.append(delta_p1)
        deltas_p2_list.append(delta_p2)

        compound_labels.append(cid)

    # ── Inter raw cosine similarity ──
    inter_cos_sims: List[float] = []
    if len(plate_means) > 1:
        plate_means_stack = torch.stack(plate_means, dim=0)  # (K, D)
        plate_means_norm = F.normalize(plate_means_stack, dim=-1)
        cos_matrix = torch.mm(plate_means_norm, plate_means_norm.T)  # (K, K)
        for j in tqdm(range(len(plate_means)), desc="Inter raw cos sim"):
            mask = torch.ones(len(plate_means), dtype=torch.bool)
            mask[j] = False
            inter_cos_sims.append(cos_matrix[j][mask].mean().item())

    # ── Delta-based cosine similarities ──
    deltas_p1_stack = torch.stack(deltas_p1_list, dim=0)  # (K, D)
    deltas_p2_stack = torch.stack(deltas_p2_list, dim=0)  # (K, D)

    intra_delta_cos_sims = F.cosine_similarity(
        deltas_p1_stack, deltas_p2_stack, dim=-1
    ).tolist()  # (K,)

    inter_delta_cos_sims: List[float] = []
    K = deltas_p1_stack.shape[0]
    if K > 1:
        mean_deltas = ((deltas_p1_stack + deltas_p2_stack) / 2.0)  # (K, D)
        mean_deltas_norm = F.normalize(mean_deltas, dim=-1)
        delta_cos_matrix = torch.mm(mean_deltas_norm, mean_deltas_norm.T)  # (K, K)
        for j in tqdm(range(K), desc="Inter delta cos sim"):
            mask = torch.ones(K, dtype=torch.bool)
            mask[j] = False
            inter_delta_cos_sims.append(delta_cos_matrix[j][mask].mean().item())

    # ── Assemble results ──
    results = {
        "num_compounds": len(compound_labels),
        "compounds": compound_labels,
        "intra_compound_cos_sim_mean": sum(intra_cos_sims) / len(intra_cos_sims),
        "intra_compound_cos_sim": dict(zip(compound_labels, intra_cos_sims)),
        "inter_compound_cos_sim_mean": (
            sum(inter_cos_sims) / len(inter_cos_sims) if inter_cos_sims else None
        ),
        "inter_compound_cos_sim": dict(zip(compound_labels, inter_cos_sims)),
        "intra_delta_cos_sim_mean": (
            sum(intra_delta_cos_sims) / len(intra_delta_cos_sims)
        ),
        "intra_delta_cos_sim": dict(zip(compound_labels, intra_delta_cos_sims)),
        "inter_delta_cos_sim_mean": (
            sum(inter_delta_cos_sims) / len(inter_delta_cos_sims)
            if inter_delta_cos_sims else None
        ),
        "inter_delta_cos_sim": dict(zip(compound_labels, inter_delta_cos_sims)),
    }
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute intra/inter cosine similarities from embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to embeddings.pt (output of encode_embeddings.py).",
    )
    parser.add_argument(
        "--compound_csv",
        type=str,
        default="Experiments/efficacy_500ppm.csv",
        help="CSV with a 'Compound No' column to select compounds.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results as JSON. If not set, prints to stdout.",
    )
    args = parser.parse_args()

    results = compute_similarities(args.embeddings, args.compound_csv)

    # Pretty-print summary
    print(f"Compounds evaluated: {results['num_compounds']}")
    print(f"Intra compound cos sim (mean): {results['intra_compound_cos_sim_mean']:.4f}")
    if results["inter_compound_cos_sim_mean"] is not None:
        print(f"Inter compound cos sim (mean): {results['inter_compound_cos_sim_mean']:.4f}")
    print(f"Intra delta cos sim   (mean): {results['intra_delta_cos_sim_mean']:.4f}")
    if results["inter_delta_cos_sim_mean"] is not None:
        print(f"Inter delta cos sim   (mean): {results['inter_delta_cos_sim_mean']:.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")
    else:
        print("\nFull results:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
