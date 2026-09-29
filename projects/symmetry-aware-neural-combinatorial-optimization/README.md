# Symmetry-Aware Neural Combinatorial Optimization

A compact symmetry laboratory adjacent to the portfolio's POMO/RL4CO routing work.

The project provides:

- the eight dihedral symmetries of square Euclidean routing instances;
- tour-cost invariance checks;
- symmetry-aware multi-start evaluation;
- a simple consistency penalty that can be attached to neural policy/value outputs across symmetric augmentations.

This isolates the core engineering idea behind symmetry-aware NCO without claiming a full reproduction of Sym-NCO. The intended next integration is to feed these augmentations and consistency terms into the existing RL4CO/POMO project, then compare plain POMO versus symmetry-aware training on matched CVRP/TSP distributions and out-of-distribution sizes.
