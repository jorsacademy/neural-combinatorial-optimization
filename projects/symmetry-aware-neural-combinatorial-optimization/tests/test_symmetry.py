import numpy as np

from sym_nco.symmetry import dihedral_augmentations, nearest_neighbor_tour, symmetry_consistency, tour_length


def test_tour_length_is_invariant_to_square_symmetries() -> None:
    rng=np.random.default_rng(2)
    coords=rng.uniform(size=(12,2))
    tour=nearest_neighbor_tour(coords,0)
    lengths=[tour_length(aug,tour) for aug in dihedral_augmentations(coords)]
    assert np.allclose(lengths,lengths[0],atol=1e-10)


def test_consistency_penalty_zero_for_equal_values() -> None:
    assert symmetry_consistency([3.0]*8) == 0.0
