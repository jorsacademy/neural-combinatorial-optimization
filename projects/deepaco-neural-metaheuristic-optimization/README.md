# DeepACO-Style Neural Metaheuristic Optimization

A compact hybrid neural + classical metaheuristic benchmark for Euclidean TSP.

The neural component does **not** construct a tour directly. A small edge-scoring network learns a heuristic prior from reference tours. Classical Ant Colony Optimization then combines that prior with inverse distance, pheromone evaporation, stochastic construction, and pheromone reinforcement.

This fills a different portfolio role from Neural LNS: learning shapes the ACO heuristic landscape while the classical metaheuristic remains responsible for search and feasibility.

## Scope boundary

This is DeepACO-inspired rather than a reproduction of a specific published architecture. The project is designed to make the interface

`learned edge prior -> classical ACO -> feasible tour`

auditable. Natural extensions are graph neural edge encoders, problem-specific local search, multiple CO problem classes, and stronger reference/teacher policies.
