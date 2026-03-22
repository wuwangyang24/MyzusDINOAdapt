"""
visualize_embeddings.py

Compare embedding clusters across different models for a chosen subset of
compounds.  Each model's embeddings are loaded from a .pt file produced by
encode_embeddings.py, projected to 2-D with UMAP (or t-SNE / PCA), and shown
side-by-side.

Output .pt file structure expected (from encode_embeddings.py):
    {
        <compound_id (str)>: {
            <plate_id (str)>: {
                "treated": torch.Tensor,   # (N, D) — one row per image
                "control": torch.Tensor,   # (D,)   — averaged across controls
            }
        }
    }

Usage examples:

    # Compare two models for 5 random compounds using UMAP (default)
    python visualize_embeddings.py \
        --embeddings  dino_base.pt  dino_lora.pt \
        --labels      "DINO Base"   "DINO+LoRA" \
        --num_compounds 5

    # Use t-SNE instead of UMAP, include control centroids
    python visualize_embeddings.py \
        --embeddings  dino_base.pt  dino_lora.pt  dino_dora.pt \
        --labels      "DINO"        "LoRA"        "DoRA" \
        --num_compounds 10 \
        --method      tsne \
        --include_controls

    # PCA, save to file instead of interactive window
    python visualize_embeddings.py \
        --embeddings  dino_base.pt  dino_lora.pt \
        --labels      "DINO"        "LoRA" \
        --num_compounds 3 \
        --method      pca \
        --output      comparison.png

    # Fit dimensionality reduction jointly across all models (shared space)
    python visualize_embeddings.py \
        --embeddings  dino_base.pt  dino_lora.pt \
        --labels      "DINO"        "LoRA" \
        --num_compounds 5 \
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


def load_embeddings(path: str) -> Dict:
    """Load a .pt embedding file produced by encode_embeddings.py."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Embedding file not found: {p}")
    return torch.load(p, map_location="cpu", weights_only=False)


def collect_vectors(
    embeddings: Dict,
    compounds: List[str],
    include_controls: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Gather per-image treated vectors (and optionally control centroids)
    for the requested compounds.

    Returns:
        vectors:      (M, D) numpy array of all collected embeddings.
        compound_ids: length-M list — compound label for each vector.
        types:        length-M list — "treated" or "control" for each vector.
    """
    all_vecs: List[np.ndarray] = []
    all_compound_ids: List[str] = []
    all_types: List[str] = []

    for cid in compounds:
        if cid not in embeddings:
            print(f"  [WARN] Compound '{cid}' not found in embedding file — skipping.")
            continue
        plates = embeddings[cid]
        for plate_id, plate_data in plates.items():
            # Treated images — each row is one image
            if "treated" in plate_data:
                t = plate_data["treated"]
                if isinstance(t, torch.Tensor):
                    t = t.numpy()
                if t.ndim == 1:
                    t = t[np.newaxis, :]
                all_vecs.append(t)
                all_compound_ids.extend([cid] * t.shape[0])
                all_types.extend(["treated"] * t.shape[0])

            # Control centroid
            if include_controls and "control" in plate_data:
                c = plate_data["control"]
                if isinstance(c, torch.Tensor):
                    c = c.numpy()
                if c.ndim == 1:
                    c = c[np.newaxis, :]
                all_vecs.append(c)
                all_compound_ids.extend([cid] * c.shape[0])
                all_types.extend(["control"] * c.shape[0])

    if not all_vecs:
        raise ValueError("No vectors found for the requested compounds.")

    vectors = np.concatenate(all_vecs, axis=0).astype(np.float32)
    return vectors, all_compound_ids, all_types


def reduce_dimensions(
    data: np.ndarray,
    method: str = "umap",
    random_state: int = 42,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> np.ndarray:
    """Project (N, D) data to (N, 2) using the chosen method."""
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


def _get_compound_color_map(compounds: List[str]) -> Dict[str, str]:
    """Assign a unique colour to each compound from a qualitative colourmap."""
    cmap = plt.get_cmap("tab20")
    unique = sorted(set(compounds))
    return {cid: cmap(i % 20) for i, cid in enumerate(unique)}


def plot_single(
    ax: plt.Axes,
    coords_2d: np.ndarray,
    compound_ids: List[str],
    types: List[str],
    color_map: Dict[str, str],
    title: str,
    include_controls: bool,
    marker_size: float = 20.0,
) -> None:
    """Scatter-plot one model's projected embeddings onto *ax*."""
    compound_arr = np.array(compound_ids)
    type_arr = np.array(types)

    for cid in sorted(set(compound_ids)):
        mask_treated = (compound_arr == cid) & (type_arr == "treated")
        if mask_treated.any():
            ax.scatter(
                coords_2d[mask_treated, 0],
                coords_2d[mask_treated, 1],
                c=[color_map[cid]],
                s=marker_size,
                alpha=0.7,
                label=f"Cpd {cid}",
                edgecolors="none",
            )

        if include_controls:
            mask_ctrl = (compound_arr == cid) & (type_arr == "control")
            if mask_ctrl.any():
                ax.scatter(
                    coords_2d[mask_ctrl, 0],
                    coords_2d[mask_ctrl, 1],
                    c=[color_map[cid]],
                    s=marker_size * 4,
                    alpha=0.9,
                    marker="X",
                    edgecolors="k",
                    linewidths=0.5,
                )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def visualize(
    embedding_paths: List[str],
    labels: List[str],
    num_compounds: Optional[int] = None,
    method: str = "umap",
    include_controls: bool = False,
    joint: bool = False,
    output: Optional[str] = None,
    perplexity: float = 30.0,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    marker_size: float = 20.0,
    figscale: float = 5.0,
) -> None:
    """
    Main visualisation routine.

    Args:
        embedding_paths: List of .pt files (one per model).
        labels:          Display name for each model.
        num_compounds:   Number of compounds to randomly sample.  If None,
                         all compounds from the first file are used.
        method:          Dimensionality-reduction method (umap | tsne | pca).
        include_controls: Show control centroids alongside treated dots.
        joint:           Fit reduction on concatenated data from all models
                         so they share the same 2-D space.
        output:          If given, save the figure to this path instead of
                         showing interactively.
        perplexity:      t-SNE perplexity.
        n_neighbors:     UMAP n_neighbors.
        min_dist:        UMAP min_dist.
        random_state:    Random seed for reproducibility.
        marker_size:     Base marker size for treated dots.
        figscale:        Scaling factor for subplot width/height.
    """
    n_models = len(embedding_paths)

    # ── Select shared random subset of compounds ─────────────────────
    # Use the first embedding file to discover available compound IDs,
    # then randomly sample the requested number.
    first_emb = load_embeddings(embedding_paths[0])
    available = sorted(first_emb.keys())
    if num_compounds is not None:
        k = min(num_compounds, len(available))
        rng = random.Random(random_state)
        compounds = sorted(rng.sample(available, k))
        print(f"Randomly selected {k} compounds: {compounds}")
    else:
        compounds = available

    # ── Collect vectors per model ────────────────────────────────────
    all_model_data: List[Tuple[np.ndarray, List[str], List[str]]] = []
    for path in embedding_paths:
        emb = load_embeddings(path)
        vecs, cids, types = collect_vectors(emb, compounds, include_controls)
        all_model_data.append((vecs, cids, types))

    # ── Dimensionality reduction ─────────────────────────────────────
    if joint:
        # Concatenate all models and project together (shared space)
        all_vecs = np.concatenate([d[0] for d in all_model_data], axis=0)
        all_coords = reduce_dimensions(
            all_vecs, method=method, random_state=random_state,
            perplexity=perplexity, n_neighbors=n_neighbors, min_dist=min_dist,
        )
        offsets = np.cumsum([0] + [d[0].shape[0] for d in all_model_data])
        coords_per_model = [
            all_coords[offsets[i]: offsets[i + 1]]
            for i in range(n_models)
        ]
    else:
        # Each model gets its own projection
        coords_per_model = []
        for vecs, _, _ in all_model_data:
            coords = reduce_dimensions(
                vecs, method=method, random_state=random_state,
                perplexity=perplexity, n_neighbors=n_neighbors, min_dist=min_dist,
            )
            coords_per_model.append(coords)

    # ── Shared colour map (same colour per compound across subplots) ─
    all_cids = []
    for _, cids, _ in all_model_data:
        all_cids.extend(cids)
    color_map = _get_compound_color_map(all_cids)

    # ── Plot ─────────────────────────────────────────────────────────
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
        vecs, cids, types = all_model_data[idx]
        plot_single(
            ax, coords_per_model[idx], cids, types,
            color_map, labels[idx], include_controls,
            marker_size=marker_size,
        )

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    # ── Shared legend ────────────────────────────────────────────────
    unique_compounds = sorted(set(all_cids))
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[c],
               markersize=8, label=f"Cpd {c}")
        for c in unique_compounds
    ]
    if include_controls:
        legend_handles.append(
            Line2D([0], [0], marker="X", color="w", markerfacecolor="grey",
                   markeredgecolor="k", markersize=10, label="Control")
        )
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
        f"Embedding Clusters — {method_label} ({mode_label})",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=200, bbox_inches="tight")
        print(f"Figure saved to: {output}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise and compare embedding clusters across models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--embeddings", type=str, nargs="+", required=True,
        help="Paths to .pt embedding files (one per model).",
    )
    parser.add_argument(
        "--labels", type=str, nargs="+", default=None,
        help="Display names for each model (same order as --embeddings). "
             "Defaults to filenames.",
    )
    parser.add_argument(
        "--num_compounds", type=int, default=None,
        help="Number of compounds to randomly sample from the first "
             "embedding file.  If omitted, all compounds are used.",
    )
    parser.add_argument(
        "--method", type=str, default="umap",
        choices=["umap", "tsne", "pca"],
        help="Dimensionality-reduction method. Default: umap.",
    )
    parser.add_argument(
        "--include_controls", action="store_true", default=False,
        help="Also plot control centroids (shown as large ✕ markers).",
    )
    parser.add_argument(
        "--joint", action="store_true", default=False,
        help="Fit dimensionality reduction jointly across all models "
             "so they share the same 2-D coordinate space.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save figure to this path (e.g. comparison.png). "
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
        "--marker_size", type=float, default=20.0,
        help="Base marker size for scatter points. Default: 20.",
    )
    parser.add_argument(
        "--figscale", type=float, default=5.0,
        help="Scale factor for subplot dimensions. Default: 5.",
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

    visualize(
        embedding_paths=args.embeddings,
        labels=labels,
        num_compounds=args.num_compounds,
        method=args.method,
        include_controls=args.include_controls,
        joint=args.joint,
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
