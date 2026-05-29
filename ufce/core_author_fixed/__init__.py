"""Theory-corrected fork of the upstream UFCE author implementation."""

from ufce.core_author_fixed.constraint_profiles import (
    AUTHOR_PUBLIC,
    CONSTRAINT_PROFILES,
    DOMAIN_ACTIONABLE,
    DOMAIN_ACTIONABLE_CHANGED_DATASETS,
    ConstraintProfileResult,
    resolve_constraint_profile,
)
from ufce.core_author_fixed.ufce import UFCE

__all__ = [
    "AUTHOR_PUBLIC",
    "CONSTRAINT_PROFILES",
    "DOMAIN_ACTIONABLE",
    "DOMAIN_ACTIONABLE_CHANGED_DATASETS",
    "ConstraintProfileResult",
    "UFCE",
    "resolve_constraint_profile",
]
