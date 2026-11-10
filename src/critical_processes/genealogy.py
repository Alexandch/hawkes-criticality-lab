"""Complete immigrant-descendant clusters, distinct from daily-bin proxies."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def _one_cluster(n, decay, max_events):
    times = [0.]
    parents = [-1]
    generations = [0]
    position = 0
    while position < len(times):
        children = np.random.poisson(n)
        if len(times) + children > max_events:
            raise RuntimeError("genealogical cluster limit reached; no truncated observation is returned")
        for _ in range(children):
            times.append(times[position] + np.random.exponential(1./decay))
            parents.append(position)
            generations.append(generations[position] + 1)
        position += 1
    return np.asarray(times), np.asarray(parents), np.asarray(generations)


@njit(cache=True)
def _sample_clusters(n, decay, replicates, seed, max_events):
    np.random.seed(seed)
    sizes = np.empty(replicates, dtype=np.int64)
    durations = np.empty(replicates)
    generations = np.empty(replicates, dtype=np.int64)
    for index in range(replicates):
        times, parents, depth = _one_cluster(n, decay, max_events)
        sizes[index] = len(times)
        durations[index] = times.max()
        generations[index] = depth.max()
    return sizes, durations, generations


@njit(cache=True)
def _seeded_cluster(n, decay, seed, max_events):
    np.random.seed(seed)
    return _one_cluster(n, decay, max_events)


def _validate(n, decay, seed, max_events):
    if not 0 <= n < 1 or not np.isfinite(decay) or decay <= 0 or not 0 <= seed < 2**32 or max_events < 1:
        raise ValueError("require 0<=n<1, positive finite decay/limit and uint32 seed")


def sample_clusters(n, decay=1., replicates=100000, seed=1, max_events=1000000):
    _validate(n, decay, seed, max_events)
    if replicates < 1:
        raise ValueError("need positive replicates")
    return _sample_clusters(float(n), float(decay), int(replicates), int(seed), int(max_events))


def simulate_genealogical_cluster(n, decay=1., seed=1, max_events=1000000):
    _validate(n, decay, seed, max_events)
    return _seeded_cluster(float(n), float(decay), int(seed), int(max_events))
