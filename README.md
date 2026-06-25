# Project 2: Sleep Architecture and Memory Consolidation
## Individual Differences in the SWS-REM Temporal Gradient and Spectral Evidence for Representational Reorganization During REM Sleep

[![Paper Preprint](https://img.shields.io/badge/Zenodo-Paper%20Preprint-87CEEB?logo=zenodo)](https://doi.org/10.5281/zenodo.20857850)

---

**Author:** Shlok Khare | UC Davis | B.S. Cognitive Science & Computer Science, Minor Psychology

---

## What This Project Is

This project uses the Sleep-EDF Expanded Cassette (SC) sub-study, a publicly available polysomnographic dataset from PhysioNet containing overnight recordings from healthy adults, to characterize the between-subject distributional structure of the within-night slow-wave sleep (SWS) to REM temporal gradient, and to test whether the oscillatory spectral signatures of waking affective valence processing persist into the sleep state theorized to consolidate emotional memories.

The pipeline maps Rechtschaffen-Kales annotations to the five-class AASM system and extracts 47 features per 30-second epoch from EEG (Fpz-Cz and Pz-Oz), horizontal EOG, and submental EMG. Features span six frequency bands (delta 0.5-4 Hz, theta 4-8 Hz, alpha 8-13 Hz, sigma 11-16 Hz, beta 13-30 Hz, gamma 30-40 Hz), time-domain statistics, spindle and slow oscillation proxies, and EOG/EMG markers. An SVM classifier (RBF kernel, class-weight balanced, leave-one-subject-out cross-validation) converts per-subject recordings into predicted hypnogram sequences. Permutation importance computed in a one-vs-rest setup identifies which spectral bands drive each stage's discriminability. The REM importance profile then compares directly against Project 1's waking affective valence importance profile, using the same frequency bands, as a test of oscillatory continuity across states.

---

## Scientific Motivation

The mean-level sleep architecture pattern (SWS dominating early cycles, REM lengthening across later cycles) appears consistently across decades of polysomnographic research. What does not appear in that literature is the between-subject distributional structure of that gradient at a scale sufficient to ask whether individual deviations from the mean are random or organized. If organized, those deviations carry theoretical information about what drives the gradient in the first place.

Two frameworks compete to explain it. The two-stage hippocampal-neocortical model predicts SWS declines as hippocampal-to-neocortical memory transfer completes across cycles, with more early SWS predicting more subsequent REM expansion as the hippocampal load discharges. The synaptic homeostasis hypothesis (SHY) predicts SWS decline tracks homeostatic discharge of synaptic potentiation accumulated during prior waking, driven by slow-wave amplitude rather than cycle count or memory content. These predictions overlap at the group level but diverge in individual subjects, especially those with non-monotonic or near-absent SWS. Between-subject variability in the gradient offers a test case neither framework has faced at sufficient sample scale.

Project 1 (Khare 2026) established that beta and gamma are the only frequency bands to formally exceed matched null distributions for cross-subject valence classification in waking EEG. REM sleep is the stage most consistently linked to emotional memory processing. If consolidation involves reactivating waking affective circuits, the spectral geometry of REM discrimination should resemble that of waking valence classification. If consolidation transforms the representation instead, the two profiles should diverge. Project 2 tests that cross-domain correspondence directly, using importance computed from the same frequency bands as Project 1.

---

## Paper

**Sleep Architecture and Memory Consolidation in Healthy Adults: Individual Differences in the SWS-REM Temporal Gradient and Spectral Evidence for Representational Reorganization During REM Sleep**
Shlok Khare, UC Davis, 2026.

Published: [https://doi.org/10.5281/zenodo.20857850](https://doi.org/10.5281/zenodo.20857850)

---

## Dataset

**Sleep-EDF Expanded (sleep-edfx)**
- Source: [physionet.org/content/sleep-edfx](https://physionet.org/content/sleep-edfx/) (free via PhysioNet)
- Format: EDF PSG and hypnogram files, Cassette (SC) sub-study
- 78 subjects aged 25-34, home ambulatory cassette recordings, recording windows approximately 24 to 150 minutes
- Signals: EEG Fpz-Cz, EEG Pz-Oz, horizontal EOG, submental EMG
- Annotations: Rechtschaffen and Kales (1968), mapped to five-class AASM (Wake, N1, N2, N3, REM)
- Final analytic sample: N=69 (SC36 excluded for missing night-1 recording; SC06, SC08, SC27, SC34, SC43, SC72, SC82 excluded for fewer than 60 clean epochs)
- 6,452 clean epochs across 47 features after artifact rejection (150 µV peak-to-peak threshold)

```
data/
├── SC/              # Raw per-subject NPZ files (76 subjects loaded)
└── SC_filtered/     # Bandpass-filtered epoch arrays
```

---

## Project Structure

```
cnsp-sleep/
├── data/
│   ├── SC/                       # Raw per-subject NPZ files
│   ├── SC_filtered/              # Bandpass-filtered epoch arrays
│   ├── features/
│   │   ├── feature_matrix.npz    # (6452, 47) clean epoch feature matrix
│   │   └── feature_names.txt     # 47 feature names in column order
│   ├── results/
│   │   ├── loso_svm.npz          # SVM LOSO per-fold results
│   │   ├── loso_rf.npz           # RF LOSO per-fold results
│   │   ├── importance.npz        # Per-stage permutation importance
│   │   ├── cycle_metrics.csv     # Per-subject cycle detection output
│   │   ├── temporal_metrics.csv  # Half-night SWS and REM proportions per subject
│   │   ├── null_rf.npy           # RF permutation null distribution (100 permutations)
│   │   ├── results_summary.json  # Aggregated classifier metrics
│   │   └── predicted_hypnograms/ # Per-subject predicted and true label arrays
│   └── metadata.csv              # Subject-level epoch and artifact counts
├── figures/                      # All paper figures (17 total, 300 dpi)
├── notebooks/
│   ├── 01_data.ipynb             # Data loading, R&K to AASM mapping, EDA
│   ├── 02_feature.ipynb          # Feature extraction across 47 features and 4 channels
│   ├── 03_analysis.ipynb         # Macrostructural and temporal analysis
│   ├── 04_model.ipynb            # LOSO classifier training and evaluation
│   ├── 05_results.ipynb          # Results compilation and figure generation
│   ├── 06_extra.ipynb            # ST sub-study comparison and supplementary analyses
│   ├── 07_fix.ipynb              # Post-review computational corrections (8 fixes)
│   └── 07.5_fix.ipynb            # Additional patches including scatter regeneration
├── src/
│   ├── cycle_detect.py           # NREM-REM cycle state machine parser
│   ├── features.py               # Welch PSD, band power, time-domain, EOG/EMG extraction
│   ├── importance.py             # Permutation importance, one-vs-rest per stage
│   ├── models.py                 # SVM, RF, LOSO loop, null distribution testing
│   └── preprocess.py             # Bandpass filtering, artifact flagging, epoch arrays
└── physionet.org/                # Downloaded Sleep-EDF Expanded source files
```

---

## Research Questions

Three research questions organized the analysis. Each is listed with its conclusion.

**RQ1:** What is the distributional structure of the within-night SWS-REM temporal gradient in 69 healthy adults, and do individual deviations from the mean half-night contrast show organized patterns with respect to first-half SWS proportion and subsequent REM expansion?

*Positive individual-differences finding. First-half SWS proportion predicts both SWS decline magnitude (r=0.770, p=1.03e-14; note algebraic dependency between predictor and composite outcome) and subsequent REM expansion (r=0.351, p=0.003; clean cross-stage cross-period pair). The gradient is directionally present in 84% of individuals (58/69 for SWS decline, 55/69 for REM increase). The SWS-REM gradient is individually structured, not a statistical average that obscures random heterogeneity.*

**RQ2:** Do beta and gamma band features show elevated importance for REM stage discrimination relative to other sleep stages and relative to the waking affective valence importance profile from Project 1, and does this importance vary between early-night and late-night REM?

*Positive directional finding under a representational reorganization framing. Spectral geometries are fully inverted across states. Theta dominates REM discrimination (importance 0.063 vs. 0.002 for beta). Beta and gamma dominate waking valence classification (both p=0.010 in Khare 2026). Beta and gamma combined importance is higher in early REM (0.01487) than late REM (0.01170), inconsistent with the prediction that late-night REM is the primary stage of high-frequency affective consolidation. Sleep consolidation operates in a spectrally distinct mode from waking affective encoding.*

**RQ3:** Do the empirical macrostructural descriptors differentiate the quantitative predictions of the two-stage hippocampal-neocortical model from those of the synaptic homeostasis hypothesis, and does the between-subject variability structure favor one framework?

*Two separate arguments targeting two different theories. (A) SHY constraint: 11/69 subjects (15.9%) show non-declining SWS with near-absent N3 throughout the recording (first-half SWS 0.063 vs. 0.321 in declining subjects, Mann-Whitney p<0.0001). SHY's homeostatic non-discharge account requires meaningful first-cycle SWS accumulation to explain a flat trajectory. These subjects had near-zero SWS from the outset, so the mechanism has nothing to discharge. (B) Two-stage directional: r=0.351 between first-half SWS and subsequent REM expansion is directionally consistent with the two-stage model's prediction that hippocampal transfer during SWS creates conditions for subsequent REM consolidation. No behavioral memory data are available to confirm the causal chain.*

---

## Results

### Classification Performance (SVM Primary, LOSO, N=69)

| Stage | F1 | Recall | Precision |
|-------|-----|--------|-----------|
| Wake  | 0.549 | 0.700 | 0.452 |
| N1    | 0.536 | 0.457 | 0.648 |
| N2    | 0.648 | 0.581 | 0.734 |
| N3    | 0.737 | 0.904 | 0.622 |
| REM   | 0.496 | 0.628 | 0.409 |
| **Macro F1** | **0.593** | | |
| **Kappa**    | **0.486** | | |

RF LOSO: macro-F1 0.559, kappa 0.483. RF permutation null: observed 0.559, null 95th percentile 0.167, p<0.001. Literature benchmarks: DeepSleepNet ~0.769, XSleepNet ~0.780 (both within-subject CV; the gap reflects LOSO strictness, not a pipeline deficit). The classifier serves as the analytical instrument for macrostructural and spectral analyses, not as a contribution to the staging literature.

### Band Importance: REM vs. Waking Valence (Project 1)

| Band  | REM importance | P1 valence accuracy | P1 null decision     |
|-------|----------------|---------------------|----------------------|
| Theta | 0.06282        | 0.495               | fail to reject H0    |
| Gamma | 0.00327        | 0.543 (p=0.010)     | reject H0            |
| Beta  | 0.00214        | 0.537 (p=0.010)     | reject H0            |
| Alpha | 0.00183        | 0.504               | fail to reject H0    |

### Primary Findings

**Finding 1 (RQ1, formally established):** First-half SWS proportion predicts subsequent REM expansion (r=0.351, p=0.003), demonstrating that the SWS-REM gradient is individually structured, not an averaging artifact. The directional pattern holds in 79.7% of individuals for REM increase (55/69).

**Finding 2 (RQ2, positive directional):** Spectral geometries are fully inverted across waking and sleep states. Theta dominates REM discrimination by an order of magnitude. Beta and gamma, the only bands to formally exceed their waking valence null distributions in Project 1, are the weakest REM discriminators. Sleep consolidation does not preserve the waking spectral architecture of affective encoding. This supports representational reorganization over reactivation-fidelity accounts.

**Finding 3 (RQ2, temporal):** Beta and gamma combined importance is higher in early REM (0.01487) than late REM (0.01170). Theta importance also declines from early to late REM (0.079 vs. 0.054). Delta importance increases in late REM (ratio 2.03), consistent with slow-wave infiltration into second-half REM. The spectral data do not support the prediction that late-night REM is the primary locus of high-frequency emotional consolidation.

**Finding 4 (RQ3 SHY constraint):** The 11 non-decreasing SWS subjects show near-absent N3 throughout the recording window (first-half SWS 0.063 vs. 0.321 in declining subjects, p<0.0001). SHY's homeostatic non-discharge account cannot apply to subjects who never accumulate meaningful SWS load to begin with.

**Finding 5 (RQ3 two-stage directional):** r=0.351 between first-half SWS and subsequent REM expansion is directionally consistent with the two-stage model's prediction that hippocampal transfer during SWS creates conditions for subsequent REM consolidation. No behavioral memory data confirm the causal chain. This argument and the SHY argument above use different results and target different theories. They do not mutually reinforce.

---

## Dependencies

```
numpy
scipy
scikit-learn
mne
matplotlib
seaborn
pingouin
jupyter
```

---

## Citation

If you use this code or build on this work, please cite:

```
Khare, S. (2026). Sleep Architecture and Memory Consolidation in Healthy Adults:
Individual Differences in the SWS-REM Temporal Gradient and Spectral Evidence
for Representational Reorganization During REM Sleep. University of California, Davis.
https://doi.org/10.5281/zenodo.20857850
```

---

## Part of a Larger Research Arc

This is Project 2 of a five-project research portfolio at the intersection of computational neuroscience, cognitive science, and AI. Project 1 (Khare 2026, [https://doi.org/10.5281/zenodo.20108739](https://doi.org/10.5281/zenodo.20108739)) established the empirical foundation for what waking affective state looks like at the neural signal level, identifying beta and gamma as the only cross-subject stable spectral signatures of valence. Project 2 tests whether those signatures persist into the sleep state theorized to consolidate emotional memories and uses the same EEG frequency bands and importance methodology to make the cross-domain comparison direct. The finding (they do not persist) constrains reactivation-fidelity accounts of sleep-dependent emotional memory processing and motivates a representational reorganization framing for subsequent work in the arc.
