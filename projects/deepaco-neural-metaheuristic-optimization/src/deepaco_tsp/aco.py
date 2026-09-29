"""Neural edge heuristic embedded inside classical ant-colony search."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import nn


def _distance_matrix(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    delta = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(delta**2, axis=2))
    np.fill_diagonal(d, np.inf)
    return d


def tour_length(coords: ArrayLike, tour: ArrayLike) -> float:
    points = np.asarray(coords, dtype=float)
    route = np.asarray(tour, dtype=int)
    nxt = np.roll(route, -1)
    return float(np.sum(np.linalg.norm(points[route] - points[nxt], axis=1)))


class NeuralHeuristic(nn.Module):
    """Small edge-scoring network used as an ACO heuristic prior."""

    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _edge_features(coords: NDArray[np.float64]) -> tuple[NDArray[np.float64], list[tuple[int, int]]]:
    d = _distance_matrix(coords)
    finite = d[np.isfinite(d)]
    scale = max(float(np.mean(finite)), 1e-8)
    edges: list[tuple[int, int]] = []
    features: list[list[float]] = []
    center = coords.mean(axis=0)
    radius = np.linalg.norm(coords - center, axis=1)
    radius_scale = max(float(np.mean(radius)), 1e-8)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            edges.append((i, j))
            features.append(
                [
                    d[i, j] / scale,
                    radius[i] / radius_scale,
                    radius[j] / radius_scale,
                ]
            )
    return np.asarray(features, dtype=float), edges


def _nearest_neighbor_tour(coords: NDArray[np.float64], start: int = 0) -> NDArray[np.int64]:
    d = _distance_matrix(coords)
    unvisited = set(range(len(coords)))
    unvisited.remove(start)
    tour = [start]
    while unvisited:
        current = tour[-1]
        nxt = min(unvisited, key=lambda j: d[current, j])
        tour.append(nxt)
        unvisited.remove(nxt)
    return np.asarray(tour, dtype=np.int64)


def train_edge_heuristic(
    instances: list[NDArray[np.float64]],
    *,
    epochs: int = 80,
    seed: int = 0,
) -> NeuralHeuristic:
    """Imitate strong short-edge priors from deterministic reference tours."""

    torch.manual_seed(seed)
    model = NeuralHeuristic()
    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for coords in instances:
        features, edges = _edge_features(np.asarray(coords, dtype=float))
        tour = _nearest_neighbor_tour(np.asarray(coords, dtype=float))
        used = {
            tuple(sorted((int(tour[k]), int(tour[(k + 1) % len(tour)]))))
            for k in range(len(tour))
        }
        labels = np.asarray([1.0 if edge in used else 0.0 for edge in edges], dtype=float)
        all_x.append(features)
        all_y.append(labels)

    x = torch.tensor(np.vstack(all_x), dtype=torch.float32)
    y = torch.tensor(np.concatenate(all_y), dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    positive = max(float(y.sum()), 1.0)
    negative = max(float(y.numel() - y.sum()), 1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive))

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
    return model.eval()


def _heuristic_matrix(coords: NDArray[np.float64], model: NeuralHeuristic | None) -> NDArray[np.float64]:
    d = _distance_matrix(coords)
    heuristic = 1.0 / np.maximum(d, 1e-8)
    heuristic[~np.isfinite(heuristic)] = 0.0
    if model is None:
        return heuristic

    features, edges = _edge_features(coords)
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.tensor(features, dtype=torch.float32))).numpy()
    learned = np.ones_like(heuristic)
    for score, (i, j) in zip(scores, edges, strict=True):
        learned[i, j] = learned[j, i] = 0.25 + float(score)
    np.fill_diagonal(learned, 0.0)
    return heuristic * learned


@dataclass(frozen=True)
class ACOResult:
    tour: NDArray[np.int64]
    length: float
    iterations: int
    ants: int


def run_aco(
    coords: ArrayLike,
    *,
    model: NeuralHeuristic | None = None,
    ants: int = 24,
    iterations: int = 60,
    alpha: float = 1.0,
    beta: float = 2.0,
    evaporation: float = 0.25,
    seed: int = 0,
) -> ACOResult:
    """Run classical pheromone ACO with an optional learned edge heuristic."""

    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 4:
        raise ValueError("coords must be an (n,2) matrix with n>=4")
    rng = np.random.default_rng(seed)
    heuristic = _heuristic_matrix(points, model)
    pheromone = np.ones((len(points), len(points)), dtype=float)
    np.fill_diagonal(pheromone, 0.0)

    best_tour: NDArray[np.int64] | None = None
    best_length = float("inf")

    for _ in range(iterations):
        tours: list[tuple[NDArray[np.int64], float]] = []
        for _ant in range(ants):
            start = int(rng.integers(0, len(points)))
            unvisited = set(range(len(points)))
            unvisited.remove(start)
            tour = [start]
            while unvisited:
                i = tour[-1]
                candidates = np.asarray(sorted(unvisited), dtype=int)
                desirability = (
                    pheromone[i, candidates] ** alpha
                    * np.maximum(heuristic[i, candidates], 1e-12) ** beta
                )
                probability = desirability / desirability.sum()
                nxt = int(rng.choice(candidates, p=probability))
                tour.append(nxt)
                unvisited.remove(nxt)
            route = np.asarray(tour, dtype=np.int64)
            length = tour_length(points, route)
            tours.append((route, length))
            if length < best_length:
                best_tour, best_length = route.copy(), length

        pheromone *= 1.0 - evaporation
        for route, length in tours:
            deposit = 1.0 / max(length, 1e-12)
            for k in range(len(route)):
                i, j = int(route[k]), int(route[(k + 1) % len(route)])
                pheromone[i, j] += deposit
                pheromone[j, i] += deposit

    assert best_tour is not None
    return ACOResult(best_tour, float(best_length), iterations, ants)
