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
INSTITUTION_FULL_NAME = "GMIoT — Greater Manchester's Institute of Technology"
INSTITUTION_REGION    = "Greater Manchester"

# ---------------------------------------------------------------------------
# Course database
# ---------------------------------------------------------------------------

COURSES_DB = os.path.join(_BASE, "gmiot.sqlite")

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
# Subject tile navigation
# Maps frontend tile label → exact ssa_label value in the courses table
# ---------------------------------------------------------------------------

SSA_MAP = {
    'Engineering':    'Engineering and Manufacturing Technologies',
    'Digital & Tech': 'Information and Communication Technology',
    'Construction':   'Construction, Planning and the Built Environment',
    'Health':         'Health, Public Services and Care',
    'Arts & Media':   'Arts, Media and Publishing',
}

# ---------------------------------------------------------------------------
# Qualification tile navigation
# Maps frontend tile label → list of qual_type values in the courses table
# ---------------------------------------------------------------------------

QUAL_FILTER_MAP = {
    'T Level':           ['T Level'],
    'Apprenticeship':    ['Apprenticeship'],
    'HNC':               ['HNC', 'HNC/HTQ', 'HTQ', 'HNC/HND'],
    'HND':               ['HND', 'HND/HTQ', 'HNC/HND'],
    'Foundation Degree': ['FdA', 'FdSc', 'CertHE', 'DipHE'],
    "Bachelor's Degree": ['BA Hons', 'BEng Hons', 'BSc Hons'],
    "Master's Degree":   ['MSc'],
    'Access to HE':      ['Access to HE'],
    'Short Course':      ['Award', 'Short Course'],
}

# ---------------------------------------------------------------------------
# Subject areas — used in prompts to tell Haiku what subjects are covered
# Each entry: (ssa_label, brief description of disciplines covered)
# ---------------------------------------------------------------------------

SUBJECT_AREAS = [
    ("Engineering and Manufacturing Technologies",
     "mechanical, electrical, manufacturing, automotive"),
    ("Information and Communication Technology",
     "software development, networking, cybersecurity, data"),
    ("Construction, Planning and the Built Environment",
     "building, civil engineering, architecture, surveying"),
    ("Health, Public Services and Care",
     "nursing, healthcare, social care"),
    ("Arts, Media and Publishing",
     "creative arts, graphic design, media production"),
    ("Business, Administration and Law",
     "business management, finance, administration"),
]
