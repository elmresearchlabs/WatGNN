"""Directed graph and message passing in plain torch.

Holds the subset of DGL's graph API that WatGNN and this SE(3)-Transformer use, with DGL's
semantics: a graph is two edge-endpoint index tensors over `num_nodes` nodes, edges are numbered in
the order they were added, and every reduction runs over the edges arriving at a node (its
destination). Torch kernels are the only compiled code involved, so the model runs on any GPU the
installed torch build supports.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


class Graph:
    """Directed graph over `num_nodes` nodes; edge i runs from `src[i]` to `dst[i]`."""

    def __init__(self, edges: Tuple[Tensor, Tensor], num_nodes: int):
        src, dst = edges
        self.src = torch.as_tensor(src, dtype=torch.int64)
        self.dst = torch.as_tensor(dst, dtype=torch.int64).to(self.src.device)
        self._num_nodes = int(num_nodes)
        self.ndata: Dict[str, Tensor] = {}
        self.edata: Dict[str, Tensor] = {}

    def num_nodes(self) -> int:
        return self._num_nodes

    def num_edges(self) -> int:
        return int(self.src.shape[0])

    def edges(self, order: str = "eid") -> Tuple[Tensor, Tensor]:
        """Edge endpoints by edge id, or sorted by source then destination for `order="srcdst"`."""
        if order == "eid":
            return self.src, self.dst
        if order == "srcdst":
            perm = torch.argsort(self._keys(), stable=True)
            return self.src[perm], self.dst[perm]
        raise ValueError(f"edge order {order!r} is not one of 'eid', 'srcdst'")

    def add_edges(self, u: Tensor, v: Tensor) -> None:
        if self.edata:
            raise ValueError("edges are added before edge features are set; this graph has "
                             f"features {sorted(self.edata)}")
        self.src = torch.cat([self.src, torch.as_tensor(u, dtype=torch.int64).to(self.src.device)])
        self.dst = torch.cat([self.dst, torch.as_tensor(v, dtype=torch.int64).to(self.src.device)])

    def _keys(self) -> Tensor:
        return self.src * self._num_nodes + self.dst

    def _lookup(self, u: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        """Edge id of each (u, v) pair and whether that edge exists. For a pair joined by more than
        one edge the id is one of them; `to_simple` leaves one edge per pair."""
        keys = self._keys()
        perm = torch.argsort(keys, stable=True)
        sorted_keys = keys[perm]
        query = (torch.as_tensor(u, dtype=torch.int64).to(keys.device) * self._num_nodes
                 + torch.as_tensor(v, dtype=torch.int64).to(keys.device))
        pos = torch.searchsorted(sorted_keys, query).clamp(max=max(len(sorted_keys) - 1, 0))
        if len(sorted_keys) == 0:
            return pos, torch.zeros_like(query, dtype=torch.bool)
        return perm[pos], sorted_keys[pos] == query

    def edge_ids(self, u: Tensor, v: Tensor) -> Tensor:
        eid, found = self._lookup(u, v)
        if not bool(found.all()):
            i = int(torch.nonzero(~found)[0])
            raise ValueError(f"no edge from node {int(u[i])} to node {int(v[i])}; "
                             f"{int((~found).sum())} of {len(found)} queried pairs are absent")
        return eid

    def has_edges_between(self, u: Tensor, v: Tensor) -> Tensor:
        return self._lookup(u, v)[1]

    def to(self, device=None, non_blocking: bool = False, **_) -> "Graph":
        out = Graph((self.src.to(device, non_blocking=non_blocking),
                     self.dst.to(device, non_blocking=non_blocking)), self._num_nodes)
        out.ndata = {k: t.to(device, non_blocking=non_blocking) for k, t in self.ndata.items()}
        out.edata = {k: t.to(device, non_blocking=non_blocking) for k, t in self.edata.items()}
        return out


def to_simple(g: Graph) -> Graph:
    """One edge per (source, destination) pair, ordered by source then destination. Node features
    are kept; edge features are not."""
    keys = torch.unique(g._keys(), sorted=True)
    out = Graph((torch.div(keys, g.num_nodes(), rounding_mode="floor"), keys % g.num_nodes()),
                g.num_nodes())
    out.ndata = dict(g.ndata)
    return out


def radius_graph(x: Tensor, r: float, self_loop: bool = False) -> Graph:
    """An edge in each direction between every pair of points within `r` of each other, the
    boundary included. Distances are computed directly rather than through a matrix product, so a
    pair at the boundary is classified from its exact separation."""
    if r <= 0:
        raise ValueError(f"radius must be positive, got {r}")
    if x.shape[0] == 0:
        raise ValueError("radius graph of an empty point set")
    dist = torch.cdist(x, x, p=2, compute_mode="donot_use_mm_for_euclid_dist")
    if not self_loop:
        dist.fill_diagonal_(r + 1)
    src, dst = torch.nonzero(dist <= r, as_tuple=True)
    return Graph((src, dst), x.shape[0])


def copy_e_sum(graph: Graph, x: Tensor) -> Tensor:
    """Sum of the features of the edges arriving at each node."""
    out = x.new_zeros((graph.num_nodes(),) + tuple(x.shape[1:]))
    return out.index_add(0, graph.dst, x)


def e_dot_v(graph: Graph, e: Tensor, v: Tensor) -> Tensor:
    """Inner product, over the last axis, of each edge's feature with its destination node's.
    The reduced axis is kept with length one."""
    return (e * v[graph.dst]).sum(dim=-1, keepdim=True)


def edge_softmax(graph: Graph, logits: Tensor) -> Tensor:
    """Softmax of each edge's logit over the edges arriving at its destination node."""
    dst = graph.dst
    shape = (graph.num_nodes(),) + tuple(logits.shape[1:])
    index = dst.view((-1,) + (1,) * (logits.dim() - 1)).expand_as(logits)
    peak = logits.new_full(shape, float("-inf")).scatter_reduce(
        0, index, logits, reduce="amax", include_self=True)
    weight = torch.exp(logits - peak[dst])
    total = weight.new_zeros(shape).index_add(0, dst, weight)
    return weight / total[dst]
