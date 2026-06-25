"""
src/preprocess.py
Bandpass filtering for Sleep-EDF epoch arrays.
Channel order: [EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental]
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

SFREQ = 100  # Hz

def _sos(lo: float, hi: float, fs: int = SFREQ, order: int = 4):
    nyq = fs / 2.0
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


# Pre-compute filter coefficients (constant across subjects)
_SOS_EEG = _sos(0.5, 40.0)   # EEG: 0.5-40 Hz
_SOS_EOG = _sos(0.5, 12.0)   # EOG: 0.5-12 Hz
_SOS_EMG = _sos(10.0, 45.0)  # EMG: 10-45 Hz (Nyquist-safe cap for 100 Hz)


def filter_epochs(epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply channel-specific bandpass filters to an epoch array.

    Parameters
    ----------
    epochs : float32 (n, 4, 3000)

    Returns
    -------
    filtered : float32 (n, 4, 3000)
    emg_rms  : float32 (n,)   per-epoch RMS of the 10-45 Hz EMG envelope
    """
    x = epochs.astype(np.float64)

    x[:, 0] = sosfiltfilt(_SOS_EEG, x[:, 0], axis=-1)  # EEG Fpz-Cz
    x[:, 1] = sosfiltfilt(_SOS_EEG, x[:, 1], axis=-1)  # EEG Pz-Oz
    x[:, 2] = sosfiltfilt(_SOS_EOG, x[:, 2], axis=-1)  # EOG

    emg_filt = sosfiltfilt(_SOS_EMG, x[:, 3], axis=-1)
    emg_rms  = np.sqrt(np.mean(emg_filt ** 2, axis=-1)).astype(np.float32)
    x[:, 3]  = emg_filt

    return x.astype(np.float32), emg_rms
