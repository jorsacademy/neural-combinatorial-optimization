"""Symmetry utilities for Euclidean routing experiments."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def dihedral_augmentations(coords: ArrayLike) -> list[NDArray[np.float64]]:
    """Return eight square symmetries after centering coordinates in [0,1]^2."""
    x=np.asarray(coords,dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("coords must have shape (n,2)")
    a=x[:,0]
    b=x[:,1]
    transforms=[
        np.column_stack([a,b]),
        np.column_stack([1-a,b]),
        np.column_stack([a,1-b]),
        np.column_stack([1-a,1-b]),
        np.column_stack([b,a]),
        np.column_stack([1-b,a]),
        np.column_stack([b,1-a]),
        np.column_stack([1-b,1-a]),
    ]
    return [np.asarray(t,dtype=float) for t in transforms]


def tour_length(coords: ArrayLike, tour: ArrayLike) -> float:
    x=np.asarray(coords,dtype=float)
    order=np.asarray(tour,dtype=int)
    route=x[order]
    nxt=np.roll(route,-1,axis=0)
    return float(np.sqrt(np.sum((route-nxt)**2,axis=1)).sum())


def nearest_neighbor_tour(coords: ArrayLike, start: int=0) -> NDArray[np.int64]:
    x=np.asarray(coords,dtype=float)
    n=len(x)
    remaining=set(range(n))
    remaining.remove(start)
    tour=[start]
    while remaining:
        last=tour[-1]
        nxt=min(remaining,key=lambda j: float(np.linalg.norm(x[last]-x[j])))
        tour.append(nxt)
        remaining.remove(nxt)
    return np.asarray(tour,dtype=np.int64)


def symmetry_multistart(coords: ArrayLike) -> tuple[NDArray[np.int64],float]:
    """Evaluate NN construction across starts and coordinate symmetries."""
    x=np.asarray(coords,dtype=float)
    best_tour=None
    best_cost=float("inf")
    for aug in dihedral_augmentations(x):
        for start in range(len(x)):
            tour=nearest_neighbor_tour(aug,start)
            cost=tour_length(aug,tour)
            if cost < best_cost:
                best_cost=cost
                best_tour=tour
    if best_tour is None:
        raise ValueError("empty instance")
    return best_tour,best_cost


def symmetry_consistency(values: ArrayLike) -> float:
    """Variance penalty used as a simple symmetry-consistency regularizer."""
    v=np.asarray(values,dtype=float)
    return float(np.mean((v-v.mean())**2))
