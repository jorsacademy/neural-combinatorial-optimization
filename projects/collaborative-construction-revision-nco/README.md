# Collaborative Construction + Revision NCO

A compact routing laboratory in which two learned policies cooperate:

```text
learned construction policy -> feasible TSP tour
                              -> learned 2-opt revision proposals
                              -> exact improvement check
                              -> improved feasible tour
```

The construction network learns edge desirability from synthetic training instances. The revision network learns the value of 2-opt edge exchanges. At inference time, learned revision scores prioritize candidate moves, while an exact tour-length check remains authoritative before any move is accepted.

This fills the construction+revision gap identified in the supplied IE/OR review and is conceptually related to collaborative-policy NCO research. It is an independent small research implementation, not a reproduction claim for a specific paper.

The project is intentionally hybrid: neural models propose; classical local search verifies feasibility and improvement.
