import torch
import numpy as np


def compute_headroom_metrics(outputs, target, selection_p=0.1, alpha=None, scale=0.4):
    """
    Compute Step 0 headroom metrics for a single test instance given multi-view logits.

    Args:
        outputs: Tensor of shape (B, C), where outputs[0] is the zero-shot view
                 and outputs[1:] are the N = B - 1 augmented views.
        target: Ground-truth class index (int or scalar Tensor).
        selection_p: Confidence selection percentile for SE (default: 0.1).
        alpha: Optional manual alpha override; if None, computed via rank rule.
        scale: Scaling factor for adaptive alpha calculation (default: 0.4).

    Returns:
        dict with per-instance accuracy indicators, predictions, and diagnostics.
    """
    y = int(target.item()) if isinstance(target, torch.Tensor) else int(target)
    logits = outputs.detach().float()
    B, _ = logits.shape
    N = B - 1
    assert N >= 1, f"Expected at least 1 augmented view, got B={B}"

    probs = logits.softmax(dim=1)
    log_probs = logits.log_softmax(dim=1)
    entropies = -(probs * log_probs).sum(dim=1)
    preds = probs.argmax(dim=1)

    # View 0 (zero-shot center-crop view)
    p0 = probs[0]
    zs_entropy = entropies[0]
    pred_zs = int(preds[0].item())
    acc_zs_view0 = float(pred_zs == y)

    # Augmented views 1..N
    aug_probs = probs[1:]
    aug_entropies = entropies[1:]
    aug_preds = preds[1:]

    # -------------------------------------------------------------------------
    # 1. Current SE Rule q
    # -------------------------------------------------------------------------
    sorted_aug_idx = torch.argsort(aug_entropies, descending=False)
    selected_num_use = max(1, int(N * selection_p))
    selected_num_se = min(N, 6)

    idx_use = sorted_aug_idx[:selected_num_use]
    idx_se = sorted_aug_idx[:selected_num_se]

    rank_ = (zs_entropy < aug_entropies).sum().item() / float(N)
    if alpha is not None:
        alpha_lin = float(alpha)
        alpha_quad = float(alpha)
    else:
        diff = rank_ - 0.5
        sign = 1.0 if diff > 0 else (-1.0 if diff < 0 else 0.0)
        alpha_lin = diff * scale + 0.5
        alpha_quad = sign * (diff * diff) * scale + 0.5

    # Standard USE SE rule (utils/use.py: linear alpha, selection_p)
    q_se_use = alpha_lin * p0 + (1.0 - alpha_lin) * aug_probs[idx_use].mean(dim=0)
    pred_q_se_use = int(q_se_use.argmax().item())
    acc_q_se_use = float(pred_q_se_use == y)

    # Variant SE rule (instance_method/se.py: quadratic alpha, top-6)
    q_se_quad = alpha_quad * p0 + (1.0 - alpha_quad) * aug_probs[idx_se].mean(dim=0)
    pred_q_se_quad = int(q_se_quad.argmax().item())
    acc_q_se_quad = float(pred_q_se_quad == y)

    # -------------------------------------------------------------------------
    # 2. Oracle q built from the subset of views predicting the true class y
    # -------------------------------------------------------------------------
    correct_mask_all = (preds == y)      # (B,)
    correct_mask_aug = (aug_preds == y)  # (N,)
    num_correct_all = int(correct_mask_all.sum().item())
    num_correct_aug = int(correct_mask_aug.sum().item())

    # 2a. Pure oracle q over all views (0..N) that predict y
    if num_correct_all > 0:
        q_oracle_pure_all = probs[correct_mask_all].mean(dim=0)
    else:
        q_oracle_pure_all = q_se_use
    pred_q_oracle_pure_all = int(q_oracle_pure_all.argmax().item())
    acc_q_oracle_pure_all = float(pred_q_oracle_pure_all == y)

    # 2b. Pure oracle q over augmented views (1..N) that predict y
    if num_correct_aug > 0:
        oracle_aug_mean = aug_probs[correct_mask_aug].mean(dim=0)
        q_oracle_pure_aug = oracle_aug_mean
    else:
        oracle_aug_mean = None
        q_oracle_pure_aug = p0
    pred_q_oracle_pure_aug = int(oracle_aug_mean.argmax().item()) if oracle_aug_mean is not None else -1
    acc_q_oracle_pure_aug = float(pred_q_oracle_pure_aug == y)

    # 2c. SE-mixed oracle q: alpha * p0 + (1 - alpha) * mean(aug_probs[S*_aug])
    # (Falls back to p0 when no augmented view predicts y)
    if oracle_aug_mean is not None:
        q_oracle_se_mixed = alpha_lin * p0 + (1.0 - alpha_lin) * oracle_aug_mean
        q_oracle_se_mixed_quad = alpha_quad * p0 + (1.0 - alpha_quad) * oracle_aug_mean
    else:
        q_oracle_se_mixed = p0
        q_oracle_se_mixed_quad = p0

    pred_q_oracle_se_mixed = int(q_oracle_se_mixed.argmax().item())
    acc_q_oracle_se_mixed = float(pred_q_oracle_se_mixed == y)

    pred_q_oracle_se_mixed_quad = int(q_oracle_se_mixed_quad.argmax().item())
    acc_q_oracle_se_mixed_quad = float(pred_q_oracle_se_mixed_quad == y)

    # -------------------------------------------------------------------------
    # 3. Best Single View
    # -------------------------------------------------------------------------
    # 3a. Lowest-entropy single augmented view
    min_ent_aug_idx = int(sorted_aug_idx[0].item())
    pred_min_ent_aug = int(aug_preds[min_ent_aug_idx].item())
    acc_min_ent_single_aug = float(pred_min_ent_aug == y)

    # 3b. Lowest-entropy single view across all views (0..N)
    min_ent_all_idx = int(torch.argsort(entropies, descending=False)[0].item())
    pred_min_ent_all = int(preds[min_ent_all_idx].item())
    acc_min_ent_single_all = float(pred_min_ent_all == y)

    # 3c. Per-instance oracle single view (1 if at least one view predicts y)
    acc_oracle_single_all = float(num_correct_all > 0)
    acc_oracle_single_aug = float(num_correct_aug > 0)

    # -------------------------------------------------------------------------
    # 4. Diagnostics
    # -------------------------------------------------------------------------
    se_precision = float(correct_mask_aug[idx_use].float().mean().item())
    aug_correct_ratio = float(num_correct_aug) / float(N)

    zs_wrong_se_rescued = float((pred_zs != y) and (pred_q_se_use == y))
    zs_wrong_oracle_mixed_rescued = float((pred_zs != y) and (pred_q_oracle_se_mixed == y))
    zs_wrong_oracle_pure_rescued = float((pred_zs != y) and (pred_q_oracle_pure_all == y))
    zs_correct_se_corrupted = float((pred_zs == y) and (pred_q_se_use != y))
    zs_correct_oracle_mixed_corrupted = float((pred_zs == y) and (pred_q_oracle_se_mixed != y))

    return {
        # Core Metric 1: SE rule q
        "acc_q_se_use": acc_q_se_use,
        "acc_q_se_quad": acc_q_se_quad,
        "pred_q_se_use": pred_q_se_use,
        # Core Metric 2: Oracle q
        "acc_q_oracle_se_mixed": acc_q_oracle_se_mixed,
        "acc_q_oracle_se_mixed_quad": acc_q_oracle_se_mixed_quad,
        "acc_q_oracle_pure_all": acc_q_oracle_pure_all,
        "acc_q_oracle_pure_aug": acc_q_oracle_pure_aug,
        # Core Metric 3: Single view
        "acc_zs_view0": acc_zs_view0,
        "acc_min_ent_single_aug": acc_min_ent_single_aug,
        "acc_min_ent_single_all": acc_min_ent_single_all,
        "acc_oracle_single_all": acc_oracle_single_all,
        "acc_oracle_single_aug": acc_oracle_single_aug,
        "per_view_correct": correct_mask_all.cpu().numpy().astype(np.float64),
        # Diagnostics
        "se_precision": se_precision,
        "aug_correct_ratio": aug_correct_ratio,
        "zs_wrong_se_rescued": zs_wrong_se_rescued,
        "zs_wrong_oracle_mixed_rescued": zs_wrong_oracle_mixed_rescued,
        "zs_wrong_oracle_pure_rescued": zs_wrong_oracle_pure_rescued,
        "zs_correct_se_corrupted": zs_correct_se_corrupted,
        "zs_correct_oracle_mixed_corrupted": zs_correct_oracle_mixed_corrupted,
        "alpha_lin": alpha_lin,
        "alpha_quad": alpha_quad,
    }


class Headroom:
    """
    Evaluator for Step 0 headroom analysis.
    Compatible with instance_tta.py's trainer interface while accumulating
    all Step 0 headroom statistics across a dataset.
    """

    SCALAR_KEYS = [
        "acc_q_se_use",
        "acc_q_se_quad",
        "acc_q_oracle_se_mixed",
        "acc_q_oracle_se_mixed_quad",
        "acc_q_oracle_pure_all",
        "acc_q_oracle_pure_aug",
        "acc_zs_view0",
        "acc_min_ent_single_aug",
        "acc_min_ent_single_all",
        "acc_oracle_single_all",
        "acc_oracle_single_aug",
        "se_precision",
        "aug_correct_ratio",
        "zs_wrong_se_rescued",
        "zs_wrong_oracle_mixed_rescued",
        "zs_wrong_oracle_pure_rescued",
        "zs_correct_se_corrupted",
        "zs_correct_oracle_mixed_corrupted",
    ]

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.reset_stats()

    def reset_stats(self):
        self.count = 0
        self.sums = {k: 0.0 for k in self.SCALAR_KEYS}
        self.per_view_correct_sum = None

    def evaluate_sample(self, images, target, args):
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
            else:
                outputs = self.model(images)

        metrics = compute_headroom_metrics(
            outputs,
            target,
            selection_p=getattr(args, "selection_p", 0.1),
            alpha=getattr(args, "alpha", None),
            scale=getattr(args, "scale", 0.4),
        )

        self.count += 1
        for k in self.SCALAR_KEYS:
            self.sums[k] += metrics[k]

        if self.per_view_correct_sum is None:
            self.per_view_correct_sum = metrics["per_view_correct"].copy()
        else:
            self.per_view_correct_sum += metrics["per_view_correct"]

        return metrics

    def adaptation_process(self, image, images, args):
        """
        Compatibility hook for instance_tta.py when target is not passed directly.
        Returns the current SE rule prediction (pred_q_se_use).
        """
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
            else:
                outputs = self.model(images)

        # Dummy target 0 when called without target; use evaluate_sample when target is known
        metrics = compute_headroom_metrics(
            outputs,
            target=0,
            selection_p=getattr(args, "selection_p", 0.1),
            alpha=getattr(args, "alpha", None),
            scale=getattr(args, "scale", 0.4),
        )
        return metrics["pred_q_se_use"]

    def summary(self):
        """Return dataset-level percentages (0-100 scale) for all headroom metrics."""
        if self.count == 0:
            return {}

        res = {"num_samples": self.count}
        for k in self.SCALAR_KEYS:
            res[k] = (self.sums[k] / float(self.count)) * 100.0

        per_view_acc = (self.per_view_correct_sum / float(self.count)) * 100.0
        res["acc_best_fixed_view_all"] = float(np.max(per_view_acc))
        res["best_fixed_view_all_idx"] = int(np.argmax(per_view_acc))
        res["acc_best_fixed_view_aug"] = float(np.max(per_view_acc[1:]))
        res["best_fixed_view_aug_idx"] = int(np.argmax(per_view_acc[1:]) + 1)
        res["acc_mean_single_aug_view"] = float(np.mean(per_view_acc[1:]))

        # Headroom gaps (percentage points)
        res["headroom_oracle_se_mixed_over_se"] = res["acc_q_oracle_se_mixed"] - res["acc_q_se_use"]
        res["headroom_oracle_pure_all_over_se"] = res["acc_q_oracle_pure_all"] - res["acc_q_se_use"]
        res["headroom_oracle_pure_aug_over_se"] = res["acc_q_oracle_pure_aug"] - res["acc_q_se_use"]
        return res

    def format_report(self, dataset_name):
        s = self.summary()
        if not s:
            return f"No samples evaluated for {dataset_name}."

        lines = [
            f"======================================================================",
            f"Step 0 Headroom Report — Dataset: [{dataset_name}] (N={s['num_samples']})",
            f"======================================================================",
            f"1. Top-1 Accuracy of q from Current SE Rule:",
            f"   - SE Rule (USE linear alpha, top-10%)        : {s['acc_q_se_use']:6.2f}%",
            f"   - SE Rule (se.py quad alpha, top-6)          : {s['acc_q_se_quad']:6.2f}%",
            f"----------------------------------------------------------------------",
            f"2. Top-1 Accuracy of Oracle q (Views Predicting True Class):",
            f"   - Oracle q (SE-mixed: alpha*p0 + (1-a)*p_S*) : {s['acc_q_oracle_se_mixed']:6.2f}%  (Headroom: +{s['headroom_oracle_se_mixed_over_se']:.2f}%)",
            f"   - Oracle q (SE-mixed quad alpha)             : {s['acc_q_oracle_se_mixed_quad']:6.2f}%",
            f"   - Oracle q (Pure subset, aug views 1..N)     : {s['acc_q_oracle_pure_aug']:6.2f}%  (Headroom: +{s['headroom_oracle_pure_aug_over_se']:.2f}%)",
            f"   - Oracle q (Pure subset, all views 0..N)     : {s['acc_q_oracle_pure_all']:6.2f}%  (Headroom: +{s['headroom_oracle_pure_all_over_se']:.2f}%)",
            f"----------------------------------------------------------------------",
            f"3. Top-1 Accuracy of Best Single View:",
            f"   - Zero-Shot View 0 (unaugmented center crop) : {s['acc_zs_view0']:6.2f}%",
            f"   - Min-Entropy Single View (aug views 1..N)   : {s['acc_min_ent_single_aug']:6.2f}%",
            f"   - Min-Entropy Single View (all views 0..N)   : {s['acc_min_ent_single_all']:6.2f}%",
            f"   - Best Fixed View Index (all 0..N, idx={s['best_fixed_view_all_idx']:2d})  : {s['acc_best_fixed_view_all']:6.2f}%",
            f"   - Best Fixed Aug View Index (1..N, idx={s['best_fixed_view_aug_idx']:2d})  : {s['acc_best_fixed_view_aug']:6.2f}%",
            f"   - Mean Single Augmented View (1..N)          : {s['acc_mean_single_aug_view']:6.2f}%",
            f"   - Per-Instance Oracle Single View (aug 1..N) : {s['acc_oracle_single_aug']:6.2f}%",
            f"   - Per-Instance Oracle Single View (all 0..N) : {s['acc_oracle_single_all']:6.2f}%",
            f"----------------------------------------------------------------------",
            f"4. View Selection Diagnostics:",
            f"   - Avg % of aug views predicting true class   : {s['aug_correct_ratio']:6.2f}%",
            f"   - SE top-K precision (% selected views == y) : {s['se_precision']:6.2f}%",
            f"   - ZS wrong -> rescued by Current SE q        : {s['zs_wrong_se_rescued']:6.2f}%",
            f"   - ZS wrong -> rescued by Oracle SE-mixed q   : {s['zs_wrong_oracle_mixed_rescued']:6.2f}%",
            f"   - ZS wrong -> rescued by Oracle Pure q       : {s['zs_wrong_oracle_pure_rescued']:6.2f}%",
            f"   - ZS right -> corrupted by Current SE q      : {s['zs_correct_se_corrupted']:6.2f}%",
            f"   - ZS right -> corrupted by Oracle SE-mixed q : {s['zs_correct_oracle_mixed_corrupted']:6.2f}%",
            f"======================================================================",
        ]
        return "\n".join(lines)

