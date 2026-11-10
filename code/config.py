"""Path configuration for the HBV-to-NAFLD transition analysis.

Raw GBD Results Tool exports and the WPP population file go in ``data/``
(see README for download parameters and target filenames). Derived tables
and figures are written to ``derived/``; text logs and intermediates to
``output/``.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
DERIVED = os.path.join(ROOT, "derived")
OUT = os.path.join(ROOT, "output")
for _d in (DATA, DERIVED, OUT):
    os.makedirs(_d, exist_ok=True)
