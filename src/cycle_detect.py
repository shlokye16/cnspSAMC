"""
src/cycle_detect.py
State-machine NREM/REM cycle detector for 30-s epoch hypnograms.
Labels: 0=Wake  1=N1  2=N2  3=N3  4=REM

Thresholds are calibrated for the Sleep-EDF Cassette dataset, which has
short recordings (mean ~70 min total sleep) and suppressed REM (6.4% raw,
mean 9 epochs/subject spread across the night). Standard clinical thresholds
(15-min NREM, 5-min REM, 3 consecutive REM for latency) produce zero
detectable cycles and undefined first-REM latency on this dataset.

Adapted values and justification:
  MIN_NREM      10 epochs  (5 min)  -- clinical: 15 min
  MIN_REM        2 epochs  (1 min)  -- clinical:  5 min
  MIN_REM_CONS   1 epoch            -- standard epidemiological latency
  MAX_WAKE_FILL 10 epochs           -- bridges brief arousals in short recordings

These preserve the essential NREM->REM architectural structure while being
scaled to the distributional reality of the dataset. Noted as a limitation
in the Methods: full cycle characterization in the clinical sense requires
longer recordings than the Cassette sub-study provides.
"""
import numpy as np
from dataclasses import dataclass

MIN_NREM      = 10   # 5 min NREM to open a cycle
MIN_REM       =  2   # 1 min REM  to close a cycle
MIN_REM_CONS  =  1   # consecutive REM epochs for first-REM latency (first occurrence)
MAX_WAKE_FILL = 10   # max inter-block Wake epochs to bridge

NREM = {1, 2, 3}


@dataclass
class Cycle:
    num:        int
    nrem_start: int
    nrem_end:   int
    rem_start:  int
    rem_end:    int
    sws_prop:   float   # N3 fraction within NREM block
    rem_dur:    int     # epochs


def _collect_block(labels: np.ndarray, i: int, target_set: set,
                   allow_fill: bool = True) -> int:
    """
    Advance i while labels[i] is in target_set, tolerating isolated
    Wake epochs that are immediately followed by a target epoch.
    Returns the exclusive end index.
    """
    n = len(labels)
    j = i
    while j < n:
        if labels[j] in target_set:
            j += 1
        elif (allow_fill and labels[j] == 0
              and j + 1 < n and labels[j + 1] in target_set):
            j += 1
        else:
            break
    return j


def detect_cycles(labels: np.ndarray) -> list[Cycle]:
    """
    Detect complete NREM/REM cycles from an epoch-label sequence.
    Dropped blocks are filtered by MIN_NREM and MIN_REM thresholds.
    """
    n      = len(labels)
    cycles = []
    c_num  = 0
    i      = 0

    while i < n:
        # Skip Wake / REM until NREM starts
        while i < n and labels[i] not in NREM:
            i += 1
        if i >= n:
            break

        # Collect NREM block (Wake gaps within NREM tolerated)
        ns = i
        ne = _collect_block(labels, i, NREM, allow_fill=True)

        if ne - ns < MIN_NREM:
            i = ne + 1
            continue

        # Bridge brief Wake between NREM end and REM start
        k = ne
        while k < n and labels[k] == 0 and k - ne < MAX_WAKE_FILL:
            k += 1

        if k >= n or labels[k] != 4:
            i = ne + 1
            continue

        # Collect REM block (Wake gaps within REM tolerated)
        rs = k
        re = _collect_block(labels, k, {4}, allow_fill=True)

        if re - rs < MIN_REM:
            i = ne + 1
            continue

        nrem_block = labels[ns:ne]
        cycles.append(Cycle(
            num=c_num, nrem_start=ns, nrem_end=ne,
            rem_start=rs, rem_end=re,
            sws_prop=float(np.sum(nrem_block == 3) / len(nrem_block)),
            rem_dur=re - rs,
        ))
        c_num += 1
        i = re

    return cycles


def first_rem_latency(labels: np.ndarray) -> int | None:
    """
    Epochs from sleep onset (first non-Wake epoch) to first REM epoch.
    Uses MIN_REM_CONS=1: the standard epidemiological definition of REM
    latency. Returns None if no REM occurs after sleep onset.
    """
    onset = next((i for i, l in enumerate(labels) if l != 0), None)
    if onset is None:
        return None
    for i in range(onset, len(labels)):
        if labels[i] == 4:
            return i - onset
    return None
