"""Accuracy and area estimation for a two-class (crop / not crop) map.

Follows Olofsson et al. (2014), "Good practices for estimating area and
assessing accuracy of land change", Remote Sensing of Environment 148:42-57.
A pixel count of the map is biased by its errors; the reference sample gives
an unbiased area estimate with a confidence interval.

Conventions: class 0 = not crop, class 1 = crop.  Error matrices are indexed
``[map class, reference class]`` and hold sample counts.
"""

from __future__ import annotations

import math

import numpy as np

CLASSES = ("not_crop", "crop")
Z95 = 1.96


def holdout_metrics(counts) -> dict:
    """Plain (unweighted) metrics for a hold-out set, rows = predicted, cols = label."""
    n = np.asarray(counts, dtype=float)
    tp, fp, fn = n[1, 1], n[1, 0], n[0, 1]
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")
    return {
        "n": int(n.sum()),
        "overall_accuracy": float(np.trace(n) / n.sum()),
        "crop_precision": float(precision),  # = user's accuracy
        "crop_recall": float(recall),  # = producer's accuracy
        "crop_f1": float(f1),
    }


def olofsson(counts, mapped_area, names=CLASSES) -> dict:
    """Stratified estimator of accuracy and area.

    counts: k x k sample counts ``[map, reference]`` from a sample stratified
    by the map classes.  mapped_area: area of each map class (any unit; the
    estimates come back in the same unit).
    """
    n = np.asarray(counts, dtype=float)
    a_map = np.asarray(mapped_area, dtype=float)
    n_i = n.sum(axis=1)
    if np.any(n_i < 2):
        raise ValueError("Each map class needs at least two reference samples")
    a_tot = a_map.sum()
    w = a_map / a_tot
    p = w[:, None] * n / n_i[:, None]  # estimated area proportions [map, ref]
    p_ref = p.sum(axis=0)

    ua = np.diag(p) / p.sum(axis=1)
    pa = np.diag(p) / p_ref
    oa = float(np.trace(p))

    frac = n / n_i[:, None]
    se_p_ref = np.sqrt(((w[:, None] ** 2) * frac * (1 - frac) / (n_i[:, None] - 1)).sum(axis=0))
    se_ua = np.sqrt(ua * (1 - ua) / (n_i - 1))
    se_oa = math.sqrt(float((w**2 * ua * (1 - ua) / (n_i - 1)).sum()))
    k = len(n)
    se_pa = np.empty(k)
    for j in range(k):  # Olofsson et al. (2014) eq. 7
        n_j = (a_map / n_i * n[:, j]).sum()  # estimated total of reference class j
        others = sum(a_map[i] ** 2 * frac[i, j] * (1 - frac[i, j]) / (n_i[i] - 1) for i in range(k) if i != j)
        se_pa[j] = math.sqrt(
            (1 / n_j**2)
            * (a_map[j] ** 2 * (1 - pa[j]) ** 2 * ua[j] * (1 - ua[j]) / (n_i[j] - 1) + pa[j] ** 2 * others)
        )

    classes = {}
    for c, name in enumerate(names):
        area, se = a_tot * p_ref[c], a_tot * se_p_ref[c]
        classes[name] = {
            "mapped_area": float(a_map[c]),
            "estimated_area": float(area),
            "area_se": float(se),
            "area_ci95": [float(area - Z95 * se), float(area + Z95 * se)],
            "users_accuracy": float(ua[c]),
            "users_accuracy_ci95": float(Z95 * se_ua[c]),
            "producers_accuracy": float(pa[c]),
            "producers_accuracy_ci95": float(Z95 * se_pa[c]),
            "samples_in_map_class": int(n_i[c]),
        }
    return {"overall_accuracy": oa, "overall_accuracy_ci95": Z95 * se_oa, "classes": classes}


def sample_allocation(
    mapped_area,
    expected_users_accuracy=(0.95, 0.75),
    target_se_overall: float = 0.01,
    min_per_class: int = 100,
) -> list[int]:
    """Reference sample size per map class (Olofsson et al. 2014, eq. 13).

    The total comes from the expected user's accuracies and the target
    standard error of overall accuracy.  Cropland is a small share of Somalia,
    so a purely proportional split would leave it a handful of points; each
    class gets at least ``min_per_class`` and the rest goes to the larger one.
    """
    a = np.asarray(mapped_area, dtype=float)
    w = a / a.sum()
    s = np.sqrt([u * (1 - u) for u in expected_users_accuracy])
    total = math.ceil((float((w * s).sum()) / target_se_overall) ** 2)
    alloc = [max(min_per_class, round(total * wk)) for wk in w]
    big = int(np.argmax(w))
    alloc[big] = max(min_per_class, total - sum(x for i, x in enumerate(alloc) if i != big))
    return alloc
