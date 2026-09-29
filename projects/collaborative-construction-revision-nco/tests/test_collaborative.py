import numpy as np

from collab_nco import construct_tour,fit_policies,revise_tour,tour_length


def test_collaborative_pipeline_returns_valid_nonworsening_tour():
    rng=np.random.default_rng(12)
    points=rng.random((9,2))
    policies=fit_policies(seed=3,instances=12,n_nodes=9,epochs=20)
    initial=construct_tour(points,policies.constructor)
    revised=revise_tour(points,initial,policies.reviser,max_rounds=8)
    assert sorted(initial.tolist())==list(range(9))
    assert sorted(revised.tolist())==list(range(9))
    assert tour_length(points,revised)<=tour_length(points,initial)+1e-9
