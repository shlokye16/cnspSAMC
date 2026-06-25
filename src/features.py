"""
src/features.py
Per-epoch feature extraction from filtered Sleep-EDF epoch arrays.
Produces a (n_epochs, n_features) matrix with named columns.
"""
import numpy as np
from scipy.signal import welch

SFREQ   = 100
NPERSEG = 256  # ~2.56 s Welch window; gives ~0.39 Hz freq resolution

# (name, lo_hz, hi_hz) - gamma included for RQ2 / Project 1 cross-domain comparison
BANDS = [
    ("delta", 0.5,  4.0),
    ("theta", 4.0,  8.0),
    ("alpha", 8.0, 13.0),
    ("sigma",11.0, 16.0),
    ("beta", 13.0, 30.0),
    ("gamma",30.0, 40.0),
    ("so",    0.5,  1.0),   # slow-oscillation subband (subset of delta)
]


def _bp(psd: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Band power via trapezoidal integration. psd: (n, F), returns (n,)."""
    mask = (freqs >= lo) & (freqs <= hi)
    return np.trapezoid(psd[:, mask], freqs[mask], axis=-1)


def _hjorth(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hjorth parameters. x: (n, samples) -> activity, mobility, complexity."""
    d1  = np.diff(x,  axis=-1)
    d2  = np.diff(d1, axis=-1)
    act = np.var(x,  axis=-1)
    mob = np.sqrt(np.var(d1, axis=-1) / (act + 1e-12))
    cmp = (np.sqrt(np.var(d2, axis=-1) / (np.var(d1, axis=-1) + 1e-12))
           / (mob + 1e-12))
    return act, mob, cmp


def extract_features(
    epochs: np.ndarray,
    emg_rms: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Parameters
    ----------
    epochs  : float32 (n, 4, 3000)  filtered channel order: EEG Fpz-Cz, Pz-Oz, EOG, EMG
    emg_rms : float32 (n,)

    Returns
    -------
    X     : float32 (n, n_features)
    names : list[str]  length n_features, same column order as X
    """
    cols:  list[np.ndarray] = []
    names: list[str]        = []

    for ci, ch in enumerate(("FpzCz", "PzOz")):
        x = epochs[:, ci].astype(np.float64)            # (n, 3000)
        freqs, psd = welch(x, fs=SFREQ, nperseg=NPERSEG, axis=-1)
        total = np.trapezoid(psd, freqs, axis=-1) + 1e-12   # (n,)

        # Spectral band features
        for bname, lo, hi in BANDS:
            absp = _bp(psd, freqs, lo, hi)
            cols  += [np.log1p(absp), absp / total]
            names += [f"{ch}_log_{bname}", f"{ch}_rel_{bname}"]

        # Spectral edge frequency 95 %
        cump = np.cumsum(psd, axis=-1)
        cump /= cump[:, -1:] + 1e-12
        sef   = freqs[np.argmax(cump >= 0.95, axis=-1)]
        cols.append(sef);  names.append(f"{ch}_sef95")

        # Delta / sigma ratio (N3 vs N2 discriminator)
        dp = _bp(psd, freqs, 0.5, 4.0)
        sp = _bp(psd, freqs, 11., 16.)
        cols.append(dp / (sp + 1e-12)); names.append(f"{ch}_delta_sigma_ratio")

        # Time-domain features
        rms  = np.sqrt(np.mean(x ** 2, axis=-1))
        ptp  = np.ptp(x, axis=-1)
        zcr  = np.sum(np.diff(np.sign(x), axis=-1) != 0, axis=-1) / x.shape[-1]
        act, mob, cmp = _hjorth(x)

        cols  += [rms, ptp, zcr, act, mob, cmp]
        names += [f"{ch}_rms", f"{ch}_ptp", f"{ch}_zcr",
                  f"{ch}_hjorth_act", f"{ch}_hjorth_mob", f"{ch}_hjorth_comp"]

    # EOG features
    eog = epochs[:, 2].astype(np.float64)
    cols  += [np.var(eog, axis=-1),
              np.sum(np.diff(np.sign(eog), axis=-1) != 0, axis=-1) / eog.shape[-1]]
    names += ["eog_var", "eog_zcr"]

    # EMG
    cols.append(emg_rms.astype(np.float64))
    names.append("emg_rms")

    X = np.column_stack(cols).astype(np.float32)
    return X, names
