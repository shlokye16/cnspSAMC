"""
src/importance.py
Per-stage permutation importance (one-vs-all) via LOSO held-out sets.
Includes the Project 1 cross-domain Spearman comparison (RQ2).
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from scipy.stats import spearmanr
from tqdm.auto import tqdm

SEED      = 42
N_STAGES  = 5
BANDS     = ["delta", "theta", "alpha", "sigma", "beta", "gamma"]


def permutation_importance_ova(
    X:       np.ndarray,
    y:       np.ndarray,
    sids:    np.ndarray,
    n_perms: int = 100,
    n_jobs:  int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-vs-all permutation importance via LOSO (RF, 200 trees for speed).

    Returns
    -------
    imp  : float64 (5, F)            mean F1 drop per stage per feature
    null : float64 (5, F, n_perms)  per-permutation drops for thresholding
    """
    subjects = np.unique(sids)
    F    = X.shape[1]
    imp  = np.zeros((N_STAGES, F))
    null = np.zeros((N_STAGES, F, n_perms))
    rng  = np.random.default_rng(SEED)

    for sid in tqdm(subjects, desc="Perm importance", unit="subj"):
        te, tr = sids == sid, sids != sid
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xte = sc.transform(X[te]).copy()
        yte = y[te]

        clf = RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=SEED, n_jobs=n_jobs)
        clf.fit(Xtr, y[tr])

        base_pred = clf.predict(Xte)
        base_f1   = np.array([
            f1_score(yte == s, base_pred == s, average="binary", zero_division=0)
            for s in range(N_STAGES)])

        for fi in range(F):
            orig = Xte[:, fi].copy()
            for pi in range(n_perms):
                Xte[:, fi] = rng.permutation(orig)
                pp = clf.predict(Xte)
                for si in range(N_STAGES):
                    drop = base_f1[si] - f1_score(
                        yte == si, pp == si, average="binary", zero_division=0)
                    imp[si, fi]      += drop
                    null[si, fi, pi] += drop
            Xte[:, fi] = orig

    imp  /= len(subjects)
    null /= len(subjects)
    return imp, null


def band_importance(imp_row: np.ndarray, feat_names: list[str]) -> dict[str, float]:
    """
    Average importance for each spectral band (across EEG channels).
    imp_row : (F,) for one stage.
    """
    return {
        b: float(np.mean([imp_row[i] for i, n in enumerate(feat_names) if b in n]))
        for b in BANDS
    }


def above_null_mask(imp: np.ndarray, null: np.ndarray, pct: float = 95.0) -> np.ndarray:
    """
    Boolean mask (5, F): True where imp exceeds the pct-th percentile of null.
    null : (5, F, n_perms)
    """
    threshold = np.percentile(null, pct, axis=-1)  # (5, F)
    return imp > threshold


def compare_project1(
    rem_imp:     np.ndarray,
    feat_names:  list[str],
    p1_band_imp: dict[str, float],
) -> tuple[float, float, list[str], list[float], list[float]]:
    """
    Spearman correlation between per-band REM importance (this paper)
    and per-band valence importance (Project 1).

    Parameters
    ----------
    rem_imp     : (F,) importance vector for REM stage
    feat_names  : feature name list
    p1_band_imp : dict mapping band name to Project 1 importance value

    Returns
    -------
    rho, pval, bands, our_vals, p1_vals
    """
    bands   = [b for b in BANDS if b in p1_band_imp]
    our_imp = [np.mean([rem_imp[i] for i, n in enumerate(feat_names) if b in n])
               for b in bands]
    p1_imp  = [p1_band_imp[b] for b in bands]
    rho, pval = spearmanr(our_imp, p1_imp)
    return rho, pval, bands, our_imp, p1_imp
