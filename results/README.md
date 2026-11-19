# Results (machine-readable)

Every estimate reported in the paper and in its supplementary appendix, as CSV, so that
values can be extracted without retyping. These are **derived results**, not the raw GBD
exports: no IHME data are redistributed here. See `README.md` in the repository root for the
exact export parameters and for the SHA-256 hash of each input file.

| File | Contents | Corresponds to |
|---|---|---|
| `table1_panelA.csv` | Global burden and the liver disease transition index (LDTI), 1990 and 2023, with relative changes | Main text Table 1, Panel A |
| `table1_panelB.csv` | Four-component Shapley decomposition of the change in the LDTI, with shared-year and independent-year 95% intervals | Main text Table 1, Panel B |
| `tableS1.csv` | Four-component decomposition by data round and period (GBD 2021 and GBD 2023) | Table S1 |
| `tableS2.csv` | NAFLD age-specific rate slopes and band-specific marginal effects, by fit window | Table S2 |
| `tableS3.csv` | Three-cause sensitivity: decomposition with hepatitis C in the denominator | Table S3 |
| `tableS4.csv` | Age-standardised sensitivity: LDTI computed from age-standardised rates | Table S4 |
| `tableS5.csv` | Reconciliation of the GBD-implied population with the UN World Population Prospects 2024 (totals, then age bands) | Table S5 |
| `tableS6.csv` | Calibration of the within-cause cross-band correlation: seed stability of the interval width, then the rho grid | Table S6 |
| `tableS7.csv` | Sensitivity of scenario C to the assumed YLD multiplier | Table S7 |
| `scenarios_2035.csv` | Projected LDTI in 2035 under scenarios A, B and C, for each of the four fit windows | Figure 4, Results 3.6 |

`tableS5.csv` and `tableS6.csv` each contain two blocks separated by a blank line (Panel A,
then Panel B), matching how the tables appear in the appendix.

Regenerating these files: the scripts in `code/` write the console logs archived in
`output/`; the tables in this folder were exported from those logs and from the analysis
outputs with the parameters given in the repository README. `w23_mc10k.py` reproduces
`table1_panelA.csv`, `table1_panelB.csv` and `tableS1.csv` byte for byte from
`data/daly_band_2023.csv`.

Contact: Xiaotong Li (20260941150@bucm.edu.cn).
