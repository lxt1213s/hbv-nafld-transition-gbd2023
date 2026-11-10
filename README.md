# Decomposing the shift from hepatitis B to non-alcoholic fatty liver disease burden (GBD 2023, 1990–2023, scenarios to 2035)

Analysis code for the manuscript under revision.

> Working title: *Decomposing the shift from hepatitis B to non-alcoholic fatty liver disease burden: a Global Burden of Disease 2023 analysis, 1990–2023, with scenarios to 2035*.

This repository contains the Python scripts that produce every quantitative result in the manuscript. Raw GBD exports and derived result tables are **not** redistributed (the GBD exports are subject to the IHME Free-of-Charge Non-Commercial User Agreement, and whether the derived tables may be released under CC BY is pending a full read of that agreement). The download parameters needed to reproduce the GBD exports are given below, so a reader can download the inputs and rerun every script.

---

## 1. Data sources

### Primary data — Global Burden of Disease Study 2023

Download from the [GBD Results Tool](https://vizhub.healthdata.org/gbd-results/). All exports are **Global only, both sexes** (the analysis is global; no country-level data are used). GBD 2023 exports were generated on **2026-11-10**; the single GBD 2021 file, used only for the round-robustness check in Table S1, was generated in **July–August 2026**.

| Target filename in `data/` | Measure | Causes | Metric | Ages | Years |
|---|---|---|---|---|---|
| `daly_band_2023.csv` | DALYs (2) | 1026 HBV, 1028 NAFLD | Number + Rate | 19 five-year bands (`<5 years` … `>95 years`) | 1990–2023 |
| `hbv_yll_yld_2015_2023.csv` | YLLs + YLDs | 1026 HBV | Number | 19 bands + `All ages` (22) | 2015, 2023 |
| `deaths_en_1980_2023.csv` | Deaths (1) | 1026 HBV, 1028 NAFLD | Number + Rate | 19 bands + `All ages` (22) + `Age-standardized` (27) | 1980–2023 |
| `deaths_zh_1980_2023.csv` | Deaths (1) | 1026 HBV, 1028 NAFLD | Number + Rate | same as above | 1980–2023 |
| `hcv_band_2023.csv` | DALYs (2) | 1027 HCV | Number + Rate | 19 five-year bands | 1990–2023 |
| `agg3c_2023.csv` | DALYs (2) | 1026 HBV, 1027 HCV, 1028 NAFLD | Number + Rate | `All ages` (22) + `Age-standardized` (27) | 1990–2023 |
| `gbd2021_global_master_1990_2021.csv` | DALYs + Deaths + Incidence | 1026 HBV, 1028 NAFLD | Number + Rate | 44 age labels incl. the 19 standard bands (script filters these 19) | 1990–2021 |

`deaths_en_…` and `deaths_zh_…` are the English- and Chinese-locale copies of the same export; only the column labels differ.

**Cause-name note.** The GBD 2023 Results Tool shows cause `1028` as *NASH-related disease burden* in the Chinese locale and *Total burden related to Non-alcoholic fatty liver disease (NAFLD)* in the English locale; the two labels refer to the same cause ID (the level-3 aggregate of NAFLD cirrhosis and liver cancer). This study keeps the GBD label "NAFLD" and does **not** equate it with the newer clinical term "MASLD".

### Population — United Nations World Population Prospects 2024

Medium variant, population by 5-year age group and sex, 1950–2100, re-binned into the 19 GBD age bands by `code/wpp_build.py` and `code/wpp_final.py` (Global only).
Source: https://population.un.org/wpp/ — file `WPP2024_Population1JanuaryByAge5GroupSex_Medium.csv.gz`, placed in `data/` as `WPP2024_pop5.gz`.

The GBD age-band population used in the decomposition is not downloaded; it is computed in-script as **DALY count ÷ rate × 100,000** (the two causes give the same population to within 1e-15). This is done inline in `dual_round.py`, `profile.py`, `w23_mc10k.py`, and `u4.py`. The WPP series is used only for the 2024–2035 scenario growth rates applied to the GBD-implied 2023 age structure.

---

## 2. Environment

Python 3.10+ with `pandas` and `numpy` (versions pinned in `requirements.txt`); no scipy required. `matplotlib` is used only by `ad2b.py` to draw the scenario figure.

---

## 3. Scripts and what each produces

`config.py` defines the three directories `data/`, `derived/`, `output/` (created on import). Run each script from the `code/` directory; it writes its log to `output/` and its result tables to `derived/` or `data/`.

| Script | Produces |
|---|---|
| `dual_round.py` | Four-component Shapley decomposition of the LDTI change, GBD 2021 and GBD 2023 rounds → `derived/dual_round_results.csv` |
| `w23_mc10k.py` | **Final** approximate 95% UI of the four components and ΔLDTI (10,000 draws, seed 20261110, ρ = 0.30, shared and independent year draws) |
| `w1_rho.py`, `n1_final.py` | Calibration of the equicorrelation ρ = 0.30 against the official GBD `All ages` intervals (`n1_final.py` is the final calibration version) |
| `ad2b.py` | Scenarios A/B/C to 2035, anchored at 2023 observed age-specific rates; scenario C splits YLL/YLD per age band (YLL moves with mortality, YLD held) |
| `af5.py` | Sensitivity on scenario C through the HBV mortality-rate exponent `s` (s = 1 is the main value) |
| `ag3.py` | Scenario C sensitivity expressed as the **HBV** YLD multiple k (k = 0, 1, 2, 3, 4); maps s to an equivalent k |
| `u4.py` | Three-cause decomposition (HBV + HCV + NAFLD denominator) and age-standardised-rate LDTI |
| `ab4.py` | 2×2 scenario contrast (HBV rate trend on/off × NAFLD rate trend on/off), four fit windows |
| `count23b.py` | Three-factor decomposition of absolute DALY counts (population growth, age-structure change, rate change) |
| `profile.py` | Crude-rate decomposition (rate/age-structure terms per 100,000) and age-profile peaks |
| `wpp_build.py`, `wpp_final.py` | Re-binning of the UN WPP 2024 population into the 19 GBD age bands (Global) |
| `check_death.py`, `echeck.py` | Data-integrity checks (duplicates, missing, lower/upper bounds, age-band coverage) |

---

## 4. Reproduction order

1. Download the GBD exports with the parameters in §1 into `data/` (filenames as in the table). Download the WPP file to `data/WPP2024_pop5.gz`.
2. `code/wpp_build.py` then `code/wpp_final.py` (generates `data/WPP2024_pop_GBDagebands_1990_2035.csv`).
3. Data-integrity checks: `code/check_death.py`, `code/echeck.py`.
4. Decomposition and intervals: `code/dual_round.py` → `code/w23_mc10k.py`.
5. ρ calibration: `code/w1_rho.py`, `code/n1_final.py`.
6. Scenarios: `code/ad2b.py` → `code/af5.py` → `code/ag3.py`.
7. Sensitivity analyses: `code/u4.py` (three-cause), `code/ab4.py` (2×2), `code/count23b.py` (count decomposition), `code/profile.py` (crude-rate split and peaks).

Interval results use `seed = 20261110`, 10,000 draws (`w23_mc10k.py`). The `derived/` and `output/` directories are run-time artifacts and are not part of the repository.

---

## 5. License

- Code: MIT License (see `LICENSE`). Copyright (c) 2026 Xiaotong Li, Cuiyan Zhu, Dehui Yin, Ye Zhu.
- Raw GBD exports and the WPP source file are the property of their respective providers and are not redistributed here. Whether the derived result tables may be released separately is pending a full read of the IHME Free-of-Charge Non-Commercial User Agreement.
