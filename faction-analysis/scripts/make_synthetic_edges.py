"""Write a synthetic planted-partition edge file for a local scale check of the Leiden child.

Each vertex belongs to one of BLOCKS equally sized blocks. Each edge joins two uniformly random members of
the same block with probability WITHIN_SHARE, and two uniformly random vertices otherwise.

Usage: python scripts/make_synthetic_edges.py OUT.npz VERTICES BLOCKS WITHIN_SHARE DEGREE SEED
"""

import sys

import numpy as np


def main() -> int:
    out = sys.argv[1]
    vertices, blocks = int(sys.argv[2]), int(sys.argv[3])
    within_share, degree, seed = float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    rng = np.random.default_rng(seed)
    edges = vertices * degree // 2
    block = np.arange(vertices) % blocks
    rng.shuffle(block)
    # Members of a block are contiguous in ``order``: the b-th block spans starts[b] to starts[b + 1].
    order = np.argsort(block, kind='stable')
    starts = np.searchsorted(block[order], np.arange(blocks + 1))
    sizes = np.diff(starts)

    first = rng.integers(0, vertices, size=edges)
    within = rng.random(edges) < within_share
    partner = rng.integers(0, vertices, size=edges)
    member = rng.integers(0, sizes[block[first]], size=edges)
    partner = np.where(within, order[starts[block[first]] + member], partner)
    keep = first != partner
    low, high = np.minimum(first[keep], partner[keep]), np.maximum(first[keep], partner[keep])
    key = np.unique(low.astype(np.int64) * vertices + high)
    weight = rng.lognormal(mean=0.0, sigma=0.6, size=len(key))
    weight *= len(weight) / weight.sum()
    np.savez(
        out,
        u=(key // vertices).astype(np.int32),
        v=(key % vertices).astype(np.int32),
        weight=weight.astype(np.float32),
        vertices=np.int64(vertices),
        block=block,
        order=order.astype(np.int32),
    )
    print(f'{out}: {vertices} vertices, {len(key)} edges, {blocks} blocks')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
