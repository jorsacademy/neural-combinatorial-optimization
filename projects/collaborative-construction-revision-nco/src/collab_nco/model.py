"""Learned construction plus learned revision proposals for Euclidean TSP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import ArrayLike,NDArray
from torch import nn


def distance_matrix(points:ArrayLike)->NDArray[np.float64]:
    p=np.asarray(points,dtype=float)
    return np.linalg.norm(p[:,None,:]-p[None,:,:],axis=2)


def tour_length(points:ArrayLike,tour:ArrayLike)->float:
    d=distance_matrix(points)
    t=np.asarray(tour,dtype=int)
    return float(sum(d[t[i],t[(i+1)%len(t)]] for i in range(len(t))))


class ConstructionPolicy(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.net=nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,1))

    def forward(self,distance:torch.Tensor)->torch.Tensor:
        return self.net(distance.reshape(-1,1)).reshape(distance.shape)


class RevisionPolicy(nn.Module):
    def __init__(self)->None:
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,24),nn.ReLU(),nn.Linear(24,1))

    def forward(self,features:torch.Tensor)->torch.Tensor:
        return self.net(features).squeeze(-1)


@dataclass
class CollaborativePolicies:
    constructor:ConstructionPolicy
    reviser:RevisionPolicy


def fit_policies(seed:int=0,instances:int=80,n_nodes:int=10,epochs:int=120)->CollaborativePolicies:
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    edge_x=[]
    edge_y=[]
    rev_x=[]
    rev_y=[]
    for _ in range(instances):
        points=rng.random((n_nodes,2))
        d=distance_matrix(points)
        iu=np.triu_indices(n_nodes,1)
        edge_x.extend(d[iu].tolist())
        edge_y.extend((-d[iu]).tolist())
        tour=rng.permutation(n_nodes)
        for i in range(n_nodes-2):
            for j in range(i+2,n_nodes-(1 if i==0 else 0)):
                a,b=tour[i],tour[(i+1)%n_nodes]
                c,e=tour[j],tour[(j+1)%n_nodes]
                feats=[d[a,b],d[c,e],d[a,c],d[b,e]]
                gain=(d[a,b]+d[c,e])-(d[a,c]+d[b,e])
                rev_x.append(feats)
                rev_y.append(gain)
    constructor=ConstructionPolicy()
    reviser=RevisionPolicy()
    opt=torch.optim.Adam(list(constructor.parameters())+list(reviser.parameters()),lr=0.02)
    ex=torch.tensor(edge_x,dtype=torch.float32)
    ey=torch.tensor(edge_y,dtype=torch.float32)
    rx=torch.tensor(np.asarray(rev_x),dtype=torch.float32)
    ry=torch.tensor(rev_y,dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss=nn.functional.mse_loss(constructor(ex),ey)+nn.functional.mse_loss(reviser(rx),ry)
        loss.backward()
        opt.step()
    return CollaborativePolicies(constructor.eval(),reviser.eval())


def construct_tour(points:ArrayLike,policy:ConstructionPolicy,start:int=0)->NDArray[np.int64]:
    p=np.asarray(points,dtype=float)
    d=distance_matrix(p)
    n=len(p)
    tour=[int(start)]
    unvisited=set(range(n))-{int(start)}
    while unvisited:
        current=tour[-1]
        candidates=np.array(sorted(unvisited),dtype=int)
        distances=torch.tensor(d[current,candidates],dtype=torch.float32)
        with torch.no_grad():
            scores=policy(distances).numpy()
        nxt=int(candidates[int(np.argmax(scores))])
        tour.append(nxt)
        unvisited.remove(nxt)
    return np.asarray(tour,dtype=np.int64)


def revise_tour(
    points:ArrayLike,
    tour:ArrayLike,
    policy:RevisionPolicy,
    *,
    max_rounds:int=20,
)->NDArray[np.int64]:
    t=np.asarray(tour,dtype=np.int64).copy()
    d=distance_matrix(points)
    n=len(t)
    for _ in range(max_rounds):
        candidates=[]
        for i in range(n-2):
            for j in range(i+2,n-(1 if i==0 else 0)):
                a,b=t[i],t[(i+1)%n]
                c,e=t[j],t[(j+1)%n]
                feats=np.array([d[a,b],d[c,e],d[a,c],d[b,e]],dtype=np.float32)
                with torch.no_grad():
                    score=float(policy(torch.tensor(feats).reshape(1,-1))[0])
                candidates.append((score,i,j))
        improved=False
        for _,i,j in sorted(candidates,reverse=True):
            candidate=t.copy()
            candidate[i+1:j+1]=candidate[i+1:j+1][::-1]
            if tour_length(points,candidate)<tour_length(points,t)-1e-10:
                t=candidate
                improved=True
                break
        if not improved:
            break
    return t
