"""Compute metrics and confidence intervals by bootstrapping predictions."""
import numpy as np
from loguru import logger
from sklearn import metrics
from tqdm import tqdm


def _compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_pred_bin: np.ndarray, target_sens=0.92
):
    """Compute metrics based on predictions.

    Parameters
    ----------
    y_true : array, shape (n_samples)
        Ground truth.

    y_pred : array, shape (n_samples)
        Ground truth.

    y_pred_bin : array, shape (n_samples)
        Ground truth.

    target_sens : float
        Target sensitivity.

    Returns
    -------
    auc : float
        ROC-AUC.

    ap : float
        Average precision score.

    se : float
        Sensitivity.

    sp : float
        Specificity.

    ppv : float
        Positive predictive value.

    npv : float
        Negative predictive value.

    thresh : float
        Threshold for target sensitvity.

    """
    auc = metrics.roc_auc_score(y_true, y_pred)
    ap = metrics.average_precision_score(y_true, y_pred)

    if y_pred_bin is None:
        _, tpr_, thresholds_ = metrics.roc_curve(y_true=y_true, y_score=y_pred)
        idx_target_se = np.where(tpr_ >= target_sens)[0][0]
        thresh = thresholds_[idx_target_se]
        y_pred_bin = (y_pred >= thresh) * 1.0
    else:
        thresh = np.nan

    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_bin).ravel()

    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return auc, ap, se, sp, ppv, npv, thresh


def compute_ci_with_bootstrap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_bin: np.ndarray = None,
    conf_level: float = 0.95,
    n_repeats: int = 1000,
    target_sens: float = 0.92,
):
    """Compute confidence intervals for metrics using bootstrapping.

    Parameters
    ----------
    y_true : array, shape (n_samples)
        Ground truth.

    y_pred : array, shape (n_samples)
        Ground truth.

    y_pred_bin : array, shape (n_samples)
        Ground truth.

    conf_level : float
        Confidence level.

    n_repeats : int
        Number of repeats for bootstrapping.

    target_sens : float
        Target sensitivity.

    Returns
    -------
    all_metrics : Dict
        Lower and upper bounds of confidence intervals for metrics.

    """
    if y_pred_bin is None:
        logger.warning(
            "y_pred_bin not specified, will compute a threshold corresponding to"
            f" {target_sens} sensitivity"
        )

    aucs_bootstrap, aps_bootstrap = [], []
    ses_bootstrap, sps_bootstrap, thresholds_bootstrap = [], [], []
    ppvs_bootstrap, npvs_bootstrap = [], []

    for _ in tqdm(range(n_repeats)):
        if n_repeats == 1:
            idx = np.arange(0, len(y_pred))
            logger.warning("You specified n_repeats=1, no bootstrap will be performed !!")
        else:
            idx = np.random.choice(range(len(y_pred)), size=len(y_pred))

        y_true_bootstrap = y_true[idx]
        y_pred_bootstrap = y_pred[idx]
        if y_pred_bin is not None:
            y_pred_bin_bootstrap = y_pred_bin[idx]
        else:
            y_pred_bin_bootstrap = None

        try:
            auc, ap, se, sp, ppv, npv, thresh = _compute_metrics(
                y_true=y_true_bootstrap,
                y_pred=y_pred_bootstrap,
                y_pred_bin=y_pred_bin_bootstrap,
                target_sens=target_sens,
            )

            aucs_bootstrap.append(auc)
            aps_bootstrap.append(ap)
            ses_bootstrap.append(se)
            sps_bootstrap.append(sp)
            ppvs_bootstrap.append(ppv)
            npvs_bootstrap.append(npv)
            thresholds_bootstrap.append(thresh)

        except Exception:
            logger.warning(
                "Not able to compute metrics because no positive cases were found in"
                + "the bootstrapped dataset."
            )

    ci_lower_bound_auc, ci_upper_bound_auc = np.nan, np.nan
    ci_lower_bound_ap, ci_upper_bound_ap = np.nan, np.nan
    ci_lower_bound_se, ci_upper_bound_se = np.nan, np.nan
    ci_lower_bound_sp, ci_upper_bound_sp = np.nan, np.nan
    ci_lower_bound_ppv, ci_upper_bound_ppv = np.nan, np.nan
    ci_lower_bound_npv, ci_upper_bound_npv = np.nan, np.nan

    if n_repeats > 1:
        ci_lower_bound_auc = np.quantile(aucs_bootstrap, 1 - conf_level)
        ci_upper_bound_auc = np.quantile(aucs_bootstrap, conf_level)
    med_auc = np.median(aucs_bootstrap)

    if n_repeats > 1:
        ci_lower_bound_ap = np.quantile(aps_bootstrap, 1 - conf_level)
        ci_upper_bound_ap = np.quantile(aps_bootstrap, conf_level)
    med_ap = np.median(aps_bootstrap)

    if n_repeats > 1:
        ci_lower_bound_se = np.quantile(ses_bootstrap, 1 - conf_level)
        ci_upper_bound_se = np.quantile(ses_bootstrap, conf_level)
    med_se = np.median(ses_bootstrap)

    if n_repeats > 1:
        ci_lower_bound_sp = np.quantile(sps_bootstrap, 1 - conf_level)
        ci_upper_bound_sp = np.quantile(sps_bootstrap, conf_level)
    med_sp = np.median(sps_bootstrap)

    if n_repeats > 1:
        ci_lower_bound_ppv = np.quantile(ppvs_bootstrap, 1 - conf_level)
        ci_upper_bound_ppv = np.quantile(ppvs_bootstrap, conf_level)
    med_ppv = np.median(ppvs_bootstrap)

    if n_repeats > 1:
        ci_lower_bound_npv = np.quantile(npvs_bootstrap, 1 - conf_level)
        ci_upper_bound_npv = np.quantile(npvs_bootstrap, conf_level)
    med_npv = np.median(npvs_bootstrap)

    if len(thresholds_bootstrap) > 0:
        ci_lower_bound_threshold = np.quantile(thresholds_bootstrap, 1 - conf_level)
        ci_upper_bound_threshold = np.quantile(thresholds_bootstrap, conf_level)
        med_threshold = np.median(thresholds_bootstrap)
    else:
        ci_lower_bound_threshold = np.nan
        ci_upper_bound_threshold = np.nan
        med_threshold = np.nan

    all_metrics = {
        "med_auc": med_auc,
        "ci_lower_bound_auc": ci_lower_bound_auc,
        "ci_upper_bound_auc": ci_upper_bound_auc,
        "med_ap": med_ap,
        "ci_lower_bound_ap": ci_lower_bound_ap,
        "ci_upper_bound_ap": ci_upper_bound_ap,
        "med_se": med_se,
        "ci_lower_bound_se": ci_lower_bound_se,
        "ci_upper_bound_se": ci_upper_bound_se,
        "med_sp": med_sp,
        "ci_lower_bound_sp": ci_lower_bound_sp,
        "ci_upper_bound_sp": ci_upper_bound_sp,
        "med_ppv": med_ppv,
        "ci_lower_bound_ppv": ci_lower_bound_ppv,
        "ci_upper_bound_ppv": ci_upper_bound_ppv,
        "med_npv": med_npv,
        "ci_lower_bound_npv": ci_lower_bound_npv,
        "ci_upper_bound_npv": ci_upper_bound_npv,
        "med_threshold": med_threshold,
        "ci_lower_bound_threshold": ci_lower_bound_threshold,
        "ci_upper_bound_threshold": ci_upper_bound_threshold,
    }
    return all_metrics
