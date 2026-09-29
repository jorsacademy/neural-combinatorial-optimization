import numpy as np

from deepaco_tsp import run_aco, tour_length, train_edge_heuristic


def _instance(seed: int, n: int = 10) -> np.ndarray:
    return np.random.default_rng(seed).uniform(size=(n, 2))


def test_aco_returns_valid_tour() -> None:
    coords = _instance(1)
    result = run_aco(coords, ants=8, iterations=8, seed=2)
    assert sorted(result.tour.tolist()) == list(range(len(coords)))
    assert np.isclose(result.length, tour_length(coords, result.tour))


def test_learned_heuristic_integrates_with_classical_aco() -> None:
    training = [_instance(seed, 9) for seed in range(4)]
    model = train_edge_heuristic(training, epochs=8, seed=3)
    coords = _instance(99, 9)
    result = run_aco(coords, model=model, ants=8, iterations=8, seed=4)
    assert sorted(result.tour.tolist()) == list(range(len(coords)))
    assert np.isfinite(result.length)
