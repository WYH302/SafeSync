"""Receiver-content metrics. Ground-truth labels are offline evaluation only."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .data import block_fraction
from .scoring import change_score
from .wire import State, Status


def roc_auc_score(truth, scores):
    """Mann-Whitney AUROC with average ranks for ties, no model fitting."""
    labels, values = np.asarray(truth, bool), np.asarray(scores)
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ranked)) + 1]
    stops = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), float)
    for start, stop in zip(starts, stops):
        ranks[order[start:stop]] = (start+1+stop)/2
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        raise ValueError("AUROC requires both classes")
    return float((ranks[labels].sum() - positive*(positive+1)/2)/(positive*negative))


def expand(values, shape, grid):
    return np.asarray(values).reshape(grid,grid).repeat(shape[0]//grid, axis=0).repeat(shape[1]//grid, axis=1)


def task_counts(prediction, truth, valid):
    pred, target = prediction[valid], truth[valid]
    return {"tp": int(np.sum(pred & target)), "fp": int(np.sum(pred & ~target)), "fn": int(np.sum(~pred & target)), "tn": int(np.sum(~pred & ~target))}


def rates(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {"f1": 2*tp / max(1, 2*tp+fp+fn), "iou": tp / max(1, tp+fp+fn)}


def evaluate(received: State, prior: State, current_features, current_image, changed, masks, threshold, minimum_cosine=.99, minimum_psnr=30., full_decoded_image=None):
    grid, shape = received.grid, changed.shape
    pred = expand(change_score(received.features, prior.features) > threshold, shape, grid)
    counts = task_counts(pred, changed, masks["known_eval_pixels"])
    all_counts = task_counts(pred, changed, np.ones(shape, bool))
    feature_usable = change_score(received.features, current_features) <= 1 - minimum_cosine
    unknown_weights = block_fraction(masks["unknown_pixels"], grid) * (shape[0]//grid) * (shape[1]//grid)
    roi_usable, roi_psnr = np.zeros(grid**2, bool), []
    original = np.asarray(current_image, dtype=np.float32)
    full_array = np.asarray(full_decoded_image, dtype=np.float32) if full_decoded_image is not None else None
    for index in range(grid**2):
        row, col = divmod(index, grid)
        y0,y1,x0,x1 = row*shape[0]//grid,(row+1)*shape[0]//grid,col*shape[1]//grid,(col+1)*shape[1]//grid
        if index in received.rois:
            decoded = np.asarray(Image.open(io.BytesIO(received.rois[index])).convert("RGB"), dtype=np.float32)
        elif full_array is not None:
            decoded = full_array[y0:y1,x0:x1]
        else:
            continue
        reference = original[y0:y1,x0:x1]
        if decoded.shape != reference.shape:
            continue
        mse = np.mean((decoded-reference)**2)
        psnr = 120. if mse == 0 else float(10*np.log10(255**2/mse))
        roi_psnr.append(psnr)
        roi_usable[index] = psnr >= minimum_psnr
    stable = expand(received.statuses == Status.CONFIRMED_STABLE, shape, grid)
    output = {**counts, **rates(counts), "all_f1": rates(all_counts)["f1"], "unknown_pixels": float(unknown_weights.sum()), "usable_unknown_feature_pixels": float(unknown_weights[feature_usable].sum()), "usable_unknown_roi_pixels": float(unknown_weights[roi_usable].sum()), "false_stable_changed_pixels": int(np.sum(stable & changed)), "changed_pixels": int(changed.sum()), "received_token_count": int(np.sum(received.statuses == Status.UPDATED)), "deferred_token_count": int(np.sum(received.statuses == Status.DEFERRED)), "stale_token_count": int(np.sum(received.statuses == Status.STALE)), "received_roi_count": len(received.rois), "roi_psnr_min_db_capped120": min(roi_psnr) if roi_psnr else None, "feature_mse": float(np.mean((received.features-current_features)**2))}
    for prefix in ("feature", "roi"):
        denominator = output["unknown_pixels"]
        output[f"unknown_{prefix}_coverage"] = output[f"usable_unknown_{prefix}_pixels"] / denominator if denominator else None
    return output


def conditional_unknown_diagnostic(change, unknown, positive, negative, n_bins=10):
    valid = positive | negative
    c, u, y = change[valid], unknown[valid], positive[valid]
    output = {"n_positive_units": int(y.sum()), "n_negative_units": int((~y).sum()), "n_bins_requested": n_bins}
    if not y.any() or y.all():
        return {**output, "auc": None, "conditional_auc": None, "bins": []}
    edges = np.unique(np.quantile(c, np.linspace(0,1,n_bins+1)))
    ids = np.digitize(c, edges[1:-1])
    strata = []
    for index in np.unique(ids):
        subset = ids == index
        npos, nneg = int(y[subset].sum()), int((~y[subset]).sum())
        if npos and nneg:
            strata.append({"bin": int(index), "n_positive": npos, "n_negative": nneg, "auc": float(roc_auc_score(y[subset], u[subset])), "pair_weight": npos*nneg})
    denom = sum(x["pair_weight"] for x in strata)
    return {**output, "auc": float(roc_auc_score(y,u)), "change_auc": float(roc_auc_score(y,c)), "score_spearman": None, "conditional_auc": sum(x["auc"]*x["pair_weight"] for x in strata)/denom if denom else None, "mixed_bin_count": len(strata), "bins": strata}
