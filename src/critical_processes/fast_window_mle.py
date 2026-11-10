from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .background_models import shaped_background_loss
from .window_features import conditional_window_data

try:
    from numba import njit
except ImportError:
    njit = lambda **kwargs: (lambda f: f)


@njit(cache=True)
def shaped_derivatives(a, n, shape, memory, counts, total, integral):
    value, ga, gn, haa, han, hnn = a * total + n * integral - total, float(total), integral, 0., 0., 0.
    for i in range(len(memory)):
        intensity = a * shape[i] + n * memory[i]
        value -= counts[i] * np.log(intensity)
        weight = counts[i] / intensity
        ga -= weight * shape[i]
        gn -= weight * memory[i]
        weight /= intensity
        haa += weight * shape[i] ** 2
        han += weight * shape[i] * memory[i]
        hnn += weight * memory[i] ** 2
    return value / total, ga / total, gn / total, haa / total, han / total, hnn / total


@njit(cache=True)
def _newton_fit(shape, memory, counts, total, integral):
    a, n = 1., 0.
    for iteration in range(80):
        loss, ga, gn, haa, han, hnn = shaped_derivatives(a, n, shape, memory, counts, total, integral)
        free_a = not ((a <= 1.00001e-8 and ga >= 0) or (a >= 5 - 1e-10 and ga <= 0))
        free_n = not ((n <= 1e-10 and gn >= 0) or (n >= .995 - 1e-10 and gn <= 0))
        projected = max(abs(ga) if free_a else 0., abs(gn) if free_n else 0.)
        if projected < 1e-7:
            return a, n, loss, haa, han, hnn, True
        determinant = haa * hnn - han * han
        if free_a and free_n and determinant > 1e-12 * max(haa * hnn, 1e-20):
            da, dn = (-hnn * ga + han * gn) / determinant, (han * ga - haa * gn) / determinant
        else:
            da = -ga / max(haa, 1e-8) if free_a else 0.
            dn = -gn / max(hnn, 1e-8) if free_n else 0.
        da = min(5., max(1e-8, a + da)) - a
        dn = min(.995, max(0., n + dn)) - n
        if ga * da + gn * dn >= -1e-18:
            da = min(5., max(1e-8, a - ga / max(haa, 1e-8))) - a if free_a else 0.
            dn = min(.995, max(0., n - gn / max(hnn, 1e-8))) - n if free_n else 0.
        direction = ga * da + gn * dn
        step, accepted = 1., False
        for _ in range(35):
            next_a, next_n = a + step * da, n + step * dn
            next_loss = shaped_derivatives(next_a, next_n, shape, memory, counts, total, integral)[0]
            if next_loss <= loss + 1e-4 * step * direction + 1e-14:
                a, n, accepted = next_a, next_n, True
                break
            step *= .5
        if not accepted:
            break
    values = shaped_derivatives(a, n, shape, memory, counts, total, integral)
    return a, n, values[0], values[3], values[4], values[5], False


def fit_fast_window(history, start, end, daily_shape=None):
    """Same conditional likelihood as the shaped MLE; Newton with SciPy fallback.

    Return n, capped curvature feature, baseline/rate, gain/event, valid flag.
    A missing fit has finite feature placeholders plus an explicit invalid flag.
    """
    memory, counts, total, integral, _ = conditional_window_data(history, start, end)
    missing = np.array([0., 2., 1. if total else 0., 0., 0.])
    if total < 3:
        return missing
    if daily_shape is None:
        shape = np.ones(len(memory))
    else:
        profile = np.asarray(daily_shape)
        values = profile[start:end]
        if not np.isfinite(values).all() or (values <= 0).any():
            return missing
        left, right = np.searchsorted(history.times, [start, end], side="left")
        shape = profile[np.floor(history.times[left:right]).astype(int)] / values.mean()
    a, n, loss, haa, han, hnn, valid = _newton_fit(shape, memory, counts, total, integral)
    if not valid:
        fit = minimize(shaped_background_loss, [1., 0.], args=(shape, memory, counts, total, integral), jac=True,
                       method="L-BFGS-B", bounds=[(1e-8, 5.), (0., .995)],
                       options={"maxiter": 250, "ftol": 1e-12, "gtol": 1e-8})
        if not fit.success or not np.isfinite(fit.fun):
            return missing
        a, n = fit.x
        loss, _, _, haa, han, hnn = shaped_derivatives(a, n, shape, memory, counts, total, integral)
    profile_information = max(1e-12, (hnn - han * han / max(haa, 1e-12)) * total)
    se = min(2., 1 / np.sqrt(profile_information))
    null_loss = shaped_derivatives(1., 0., shape, memory, counts, total, integral)[0]
    return np.array([n, se, a, max(0., null_loss - loss), 1.])
