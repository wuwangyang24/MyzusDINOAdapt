"""
shuffle_controls.py

Load an embeddings.pt file (output of encode_embeddings.py) and shuffle
the control embeddings across plates.  All control vectors are collected
into a single pool and randomly reassigned, so each plate ends up with
a control embedding that originally belonged to a different plate.

This is useful as a negative-control experiment to verify that the delta
vectors (treated − control) carry meaningful plate-specific information.

Usage:
    python Experiments/shuffle_controls.py \
        --embeddings /path/to/embeddings.pt \
        --output     /path/to/embeddings_shuffled.pt \
        --seed       42
"""

import argparse
import copy
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np


def shuffle_controls(
    embeddings: Dict,
    seed: int = 42,
) -> Dict:
    """
    Shuffle control embeddings across all plates in the embeddings dict.

    Every (compound, plate) that has a "control" key participates in the
    shuffle.  The control vectors are randomly permuted and reassigned so
    that each plate receives a control from a different plate.

    Parameters
    ----------
    embeddings : dict
        Nested dict: compound_id -> plate_id -> {"treated": Tensor, "control": Tensor}.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Deep copy of *embeddings* with control vectors shuffled.
    """
    # Collect all (compound, plate) keys that have a control embedding
    keys: List[Tuple[str, str]] = []
    controls: List[torch.Tensor] = []

    for cid, plates in embeddings.items():
        for pid, data in plates.items():
            if "control" in data:
                keys.append((cid, pid))
                controls.append(data["control"])

    if len(controls) == 0:
        raise ValueError("No control embeddings found in the embeddings file.")

    # Shuffle the control vectors
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(controls))
    shuffled_controls = [controls[i] for i in perm]

    # Build output with shuffled controls
    result = copy.deepcopy(embeddings)
    for (cid, pid), ctrl in zip(keys, shuffled_controls):
        result[cid][pid]["control"] = ctrl

    # Report how many controls stayed in place vs moved
    n_same = sum(1 for i, j in enumerate(perm) if i == j)
    print(f"Shuffled {len(controls)} control embeddings across plates.")
    print(f"  {len(controls) - n_same} moved, {n_same} stayed in place (by chance).")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle control embeddings across plates in an embeddings.pt file."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Path to input embeddings.pt (output of encode_embeddings.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the shuffled embeddings.pt.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()

    print(f"Loading embeddings: {args.embeddings}")
    embeddings = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    print(f"  {len(embeddings)} compounds loaded.")

    result = shuffle_controls(embeddings, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out_path)
    print(f"Saved shuffled embeddings to {out_path}")


if __name__ == "__main__":
    main()
