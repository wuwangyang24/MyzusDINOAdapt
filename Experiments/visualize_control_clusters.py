"""
visualize_control_clusters.py

Visualise control-embedding clusters coloured by plate.

Loads a .pt embedding file produced by encode_embeddings.py and projects the
per-plate control centroids to 2-D (UMAP, t-SNE, or PCA) so you can inspect
how well controls cluster across plates.

Input .pt file structure (from encode_embeddings.py):
    {
        <compound_id (str)>: {
            <plate_id (str)>: {
                "treated": torch.Tensor,   # (N, D)
                "control": torch.Tensor,   # (D,)
            }
        }
    }

Usage examples:

    # Default: UMAP projection of all control centroids, coloured by plate
    python visualize_control_clusters.py \
        --embeddings /path/to/embeddings.pt

    # Compare controls from two different model runs side-by-side
    python visualize_control_clusters.py \
        --embeddings dino_base.pt dino_lora.pt \
        --labels     "DINO Base"  "DINO+LoRA"

    # Use t-SNE, save to file
    python visualize_control_clusters.py \
        --embeddings /path/to/embeddings.pt \
        --method     tsne \
        --output     control_clusters.png

    # PCA, annotate each point with compound ID
    python visualize_control_clusters.py \
        --embeddings /path/to/embeddings.pt \
        --method     pca \
        --annotate

    # Randomly sample 5 plates from the available plates
    python visualize_control_clusters.py \
        --embeddings /path/to/embeddings.pt \
        --num_plates 5

    # Joint projection across multiple embedding files
    python visualize_control_clusters.py \
        --embeddings dino_base.pt dino_lora.pt \
        --labels     "DINO Base"  "DINO+LoRA" \
        --joint
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_embeddings(path: str) -> Dict:
    """Load a .pt embedding file produced by encode_embeddings.py."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def collect_control_vectors(
    embeddings: Dict,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract all per-plate control centroid vectors.

    Returns:
        vectors:      (M, D) numpy array – one row per plate control centroid.
        plate_ids:    length-M list – plate label for each vector.
        compound_ids: length-M list – compound label for each vector.
    """
    all_vecs: List[np.ndarray] = []
    all_plate_ids: List[str] = []
    all_compound_ids: List[str] = []

    for compound_id, plates in embeddings.items():
        for plate_id, plate_data in plates.items():
            if "control" not in plate_data:
                continue
            ctrl = plate_data["control"]
            if isinstance(ctrl, torch.Tensor):
                ctrl = ctrl.numpy()
            if ctrl.ndim == 1:
                ctrl = ctrl[np.newaxis, :]          # (1, D)
            all_vecs.append(ctrl)
            all_plate_ids.extend([str(plate_id)] * ctrl.shape[0])
            all_compound_ids.extend([str(compound_id)] * ctrl.shape[0])

    if not all_vecs:
        raise ValueError("No control embeddings found in the file.")

    vectors = np.concatenate(all_vecs, axis=0).astype(np.float32)
    return vectors, all_plate_ids, all_compound_ids


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def reduce_dimensions(
    data: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project (N, D) data to (N, 2) via the chosen method."""
    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        return reducer.fit_transform(data)

    if method == "tsne":
        effective_perp = min(perplexity, max(1.0, data.shape[0] - 1))
        reducer = TSNE(
            n_components=2,
            perplexity=effective_perp,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        )
        return reducer.fit_transform(data)

    if method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for --method umap.  "
                "Install it:  pip install umap-learn"
            )
        effective_nn = min(n_neighbors, data.shape[0] - 1)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_nn,
            min_dist=min_dist,
            random_state=random_state,
        )
        return reducer.fit_transform(data)

    raise ValueError(f"Unknown method '{method}'. Choose from: umap, tsne, pca.")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _get_plate_color_map(plate_ids: List[str]) -> Dict[str, np.ndarray]:
    """Assign a unique colour to each plate from a qualitative colourmap."""
    unique = sorted(set(plate_ids))
    cmap = plt.get_cmap("tab20" if len(unique) <= 20 else "gist_ncar")
    return {pid: cmap(i / max(len(unique) - 1, 1)) for i, pid in enumerate(unique)}


def plot_single(
    ax: plt.Axes,
    coords_2d: np.ndarray,
    plate_ids: List[str],
    compound_ids: List[str],
    color_map: Dict[str, np.ndarray],
    title: str,
    marker_size: float = 60.0,
    annotate: bool = False,
) -> None:
    """Scatter-plot control centroids on *ax*, coloured by plate."""
    plate_arr = np.array(plate_ids)

    for pid in sorted(set(plate_ids)):
        mask = plate_arr == pid
        ax.scatter(
            coords_2d[mask, 0],
            coords_2d[mask, 1],
            c=[color_map[pid]],
            s=marker_size,
            alpha=0.8,
            label=f"Plate {pid}",
            edgecolors="k",
            linewidths=0.4,
        )

    if annotate:
        for i, (x, y) in enumerate(coords_2d):
            ax.annotate(
                compound_ids[i],
                (x, y),
                fontsize=6,
                alpha=0.7,
                textcoords="offset points",
                xytext=(4, 4),
            )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


# ---------------------------------------------------------------------------
# Main visualisation routine
# ---------------------------------------------------------------------------

def visualize_controls(
    embedding_paths: List[str],
    labels: List[str],
    method: str = "umap",
    joint: bool = False,
    annotate: bool = False,
    num_plates: Optional[int] = None,
    output: Optional[str] = None,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    marker_size: float = 60.0,
    figscale: float = 6.0,
) -> None:
    """
    Load embedding files, collect control centroids, reduce to 2-D, and plot.

    Args:
        embedding_paths: List of .pt files (one per model/run).
        labels:          Display name for each embedding file.
        method:          Dimensionality-reduction method (umap | tsne | pca).
        joint:           Fit reduction jointly across all files (shared space).
        annotate:        Annotate each point with its compound ID.
        num_plates:      Randomly sample this many plates. If None, use all.
        output:          Save figure to path; if None, show interactively.
        perplexity:      t-SNE perplexity.
        n_neighbors:     UMAP n_neighbors.
        min_dist:        UMAP min_dist.
        random_state:    Random seed.
        marker_size:     Scatter marker size.
        figscale:        Scale factor for subplot dimensions.
    """
    n_models = len(embedding_paths)

    # ── Collect control vectors per file ─────────────────────────────
    print(f"\n[1/5] Loading {n_models} embedding file(s)...")
    all_model_data: List[Tuple[np.ndarray, List[str], List[str]]] = []
    for path in embedding_paths:
        print(f"  Loading: {Path(path).name}")
        emb = load_embeddings(path)
        vecs, pids, cids = collect_control_vectors(emb)
        print(f"    → {vecs.shape[0]} control centroids, "
              f"dim={vecs.shape[1]}, plates={len(set(pids))}")
        all_model_data.append((vecs, pids, cids))

    # ── Optionally sub-sample plates ─────────────────────────────────
    if num_plates is not None:
        print(f"\n[2/5] Sampling {num_plates} plates...")
        # Discover the union of plate IDs across all files
        all_available_plates = sorted(
            set(pid for _, pids, _ in all_model_data for pid in pids)
        )
        k = min(num_plates, len(all_available_plates))
        rng = random.Random(random_state)
        sampled_plates = set(rng.sample(all_available_plates, k))
        print(f"  Available plates: {len(all_available_plates)}")
        print(f"  Sampled {k} plates: {sorted(sampled_plates)}")

        filtered: List[Tuple[np.ndarray, List[str], List[str]]] = []
        for vecs, pids, cids in all_model_data:
            mask = np.array([p in sampled_plates for p in pids])
            filtered.append((
                vecs[mask],
                [p for p, m in zip(pids, mask) if m],
                [c for c, m in zip(cids, mask) if m],
            ))
        all_model_data = filtered
    else:
        print(f"\n[2/5] Plate sampling: skipped (using all plates)")

    # ── Dimensionality reduction ─────────────────────────────────────
    total_points = sum(d[0].shape[0] for d in all_model_data)
    mode_str = "jointly" if joint else "independently"
    print(f"\n[3/5] Reducing dimensions with {method.upper()} ({mode_str})...")
    print(f"  Total points: {total_points}")
    reduce_kw = dict(
        method=method,
        random_state=random_state,
        perplexity=perplexity,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )

    if joint:
        combined = np.concatenate([d[0] for d in all_model_data], axis=0)
        all_coords = reduce_dimensions(combined, **reduce_kw)
        offsets = np.cumsum([0] + [d[0].shape[0] for d in all_model_data])
        coords_per_model = [
            all_coords[offsets[i]: offsets[i + 1]]
            for i in range(n_models)
        ]
    else:
        coords_per_model = [
            reduce_dimensions(vecs, **reduce_kw)
            for vecs, _, _ in all_model_data
        ]

    print(f"  ✓ Dimensionality reduction complete.")

    # ── Shared plate colour map ──────────────────────────────────────
    all_plates: List[str] = []
    for _, pids, _ in all_model_data:
        all_plates.extend(pids)
    color_map = _get_plate_color_map(all_plates)

    # ── Plot ─────────────────────────────────────────────────────────
    print(f"\n[4/5] Plotting {len(set(all_plates))} unique plates across {n_models} subplot(s)...")
    ncols = min(n_models, 3)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figscale * ncols, figscale * nrows),
        squeeze=False,
    )

    for idx in range(n_models):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        vecs, pids, cids = all_model_data[idx]
        plot_single(
            ax, coords_per_model[idx], pids, cids,
            color_map, labels[idx],
            marker_size=marker_size,
            annotate=annotate,
        )

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # ── Legend (one entry per plate) ─────────────────────────────────
    unique_plates = sorted(set(all_plates))
    legend_handles = [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=color_map[pid],
            markeredgecolor="k", markersize=8,
            label=f"Plate {pid}",
        )
        for pid in unique_plates
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(legend_handles), 8),
        fontsize=9,
        frameon=True,
        bbox_to_anchor=(0.5, -0.02),
    )

    method_label = {"umap": "UMAP", "tsne": "t-SNE", "pca": "PCA"}[method]
    mode_label = "joint" if joint else "independent"
    fig.suptitle(
        f"Control Embedding Clusters by Plate — {method_label} ({mode_label})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    print(f"\n[5/5] Saving / displaying figure...")
    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"  ✓ Figure saved to: {output}")
    else:
        print(f"  Opening interactive window...")
        plt.show()

    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise control-embedding clusters coloured by plate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--embeddings", type=str, nargs="+", required=True,
        help="Paths to .pt embedding files (one or more).",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Display names for each file (same order as --embeddings). "
             "Defaults to filenames.",
    )
    parser.add_argument(
        "--method", type=str, default="umap",
        choices=["umap", "tsne", "pca"],
        help="Dimensionality-reduction method. Default: umap.",
    )
    parser.add_argument(
        "--num_plates", type=int, default=None,
        help="Randomly sample this many plates from the available plates. "
             "If omitted, all plates are used.",
    )
    parser.add_argument(
        "--joint", action="store_true", default=False,
        help="Fit dimensionality reduction jointly across all files.",
    )
    parser.add_argument(
        "--annotate", action="store_true", default=False,
        help="Annotate each point with its compound ID.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save figure to this path (e.g. controls.png). "
             "If omitted, opens an interactive window.",
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="t-SNE perplexity. Default: 30.",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=15,
        help="UMAP n_neighbors. Default: 15.",
    )
    parser.add_argument(
        "--min_dist", type=float, default=0.1,
        help="UMAP min_dist. Default: 0.1.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--marker_size", type=float, default=60.0,
        help="Scatter marker size. Default: 60.",
    )
    parser.add_argument(
        "--figscale", type=float, default=6.0,
        help="Scale factor for subplot dimensions. Default: 6.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    labels = args.labels
    if labels is None:
        labels = [Path(p).stem for p in args.embeddings]
    if len(labels) != len(args.embeddings):
        print(f"ERROR: --labels count ({len(labels)}) must match "
              f"--embeddings count ({len(args.embeddings)}).")
        sys.exit(1)

    visualize_controls(
        embedding_paths=args.embeddings,
        labels=labels,
        method=args.method,
        joint=args.joint,
        annotate=args.annotate,
        num_plates=args.num_plates,
        output=args.output,
        perplexity=args.perplexity,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        random_state=args.seed,
        marker_size=args.marker_size,
        figscale=args.figscale,
    )


if __name__ == "__main__":
    main()
