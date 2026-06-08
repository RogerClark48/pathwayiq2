"""
institution_config.py

All institution-specific configuration for a PathwayIQ deployment.
Swap this file (or point to a different one) to deploy for a different institution.

api.py imports everything from here — no institution-specific values should be
hardcoded in api.py.
"""

import os

_BASE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

INSTITUTION_NAME      = "GMIoT"
INSTITUTION_FULL_NAME = "Greater Manchester Institute of Technology"
INSTITUTION_REGION    = "Greater Manchester"

# ---------------------------------------------------------------------------
# Course database
# ---------------------------------------------------------------------------

COURSES_DB = os.path.join(_BASE, "futurefinder.sqlite")

# ---------------------------------------------------------------------------
# Partner providers
# Key: display name  Value: location / note (used in prompts)
# ---------------------------------------------------------------------------

PROVIDERS = {
    "Wigan & Leigh College":       "Wigan",
    "University of Salford":       "Salford",
    "Trafford & Stockport College":"campuses in Stretford and Stockport",
    "Tameside College":            "Ashton-under-Lyne",
    "Bury College":                "Bury",
    "Ada College":                 "Manchester city centre; specialises in digital and technology",
}

# ---------------------------------------------------------------------------
# Qualification tile navigation
# Maps frontend tile label → list of qual_type values in the courses table
# ---------------------------------------------------------------------------

QUAL_FILTER_MAP = {
    'T Level':                  ['T Level'],
    'Access to HE':             ['Access to HE'],
    'Apprenticeship':           ['Apprenticeship'],
    'Higher Apprenticeship':    ['Higher Apprenticeship',
                                 'Higher Apprenticeship, Degree Apprenticeship',
                                 'FdSc, Higher Apprenticeship'],
    'Degree Apprenticeship':    ['Degree Apprenticeship',
                                 'Degree Apprenticeship, BSc Hons'],
    'CertHE / DipHE':           ['CertHE', 'DipHE'],
    'HNC':                      ['HNC', 'HTQ', 'HTQ, HNC', 'HNC/HTQ', 'HNC/HND'],
    'HND':                      ['HND', 'HND, HTQ', 'HND/HTQ', 'HNC/HND'],
    'Foundation Degree':        ['FdA', 'FdSc'],
    "Bachelor's Degree":        ['BA Hons', 'BEng Hons', 'BSc Hons'],
    "Master's Degree":          ['MSc', 'MSc, PgDip'],
    'Short Course':             ['Award', 'Short Course'],
}

