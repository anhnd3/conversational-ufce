"""Frozen configuration for the UPV-2025 binary evaluation."""

from dataclasses import dataclass


ZENODO_RECORD = "https://zenodo.org/records/17239943"
ZENODO_API = "https://zenodo.org/api/records/17239943"
EXPECTED_FILES = {
    "2018": {
        "md5": "43fc067c88f09797dacbcd83ee83cc28",
        "rows": 152446,
    },
    "2021": {
        "md5": "50cfb2d0f0a07bf3460bc259ffa84064",
        "rows": 153120,
    },
    "2022": {
        "md5": "b5f724fde2c33961f07f26e8fa25b4b7",
        "rows": 159173,
    },
}
EXPECTED_TOTAL_ROWS = 464739
EXPECTED_STUDENTS = 39364

STUDENT_ID = "dni_hash"
TARGET = "abandono_hash"
SOURCE_CLASS = 0
DESIRED_CLASS = 1
RANDOM_SEED = 42
PRIMARY_SEED = 0
ROBUSTNESS_SEEDS = (1, 2, 3, 4)
TIMEOUT_SECONDS = 60.0
MAX_CANDIDATES = 5

CATEGORICAL_FEATURES = [
    "tipo_ingreso",
    "campus_hash",
    "estudios_p_hash",
    "estudios_m_hash",
    "dedicacion",
    "desplazado_hash",
]

# These fields identify entities, encode the outcome, or are recorded after the
# intervention/outcome cutoff.  The source has a few version-specific omissions;
# dropping only present fields is deliberate and audited by the schema check.
DROP_FIELDS = [
    "dni_hash",
    "tit_hash",
    "asi_hash",
    "grupos_por_tipocredito_hash",
    "abandono_hash",
    "baja_fecha",
    "matricula_activa",
    "nota_asig_hash",
    "fecha_datos",
    "cred_normal",
    "cred_sup_normal",
    "cred_sup_espec",
    "cred_sup",
    "cred_sup_sem_a",
    "cred_sup_sem_b",
    "cred_sup_anu",
    "cred_sup_total",
    "rendimiento_cuat_a",
    "rendimiento_cuat_b",
    "rendimiento_total",
    "exento_npp",
    "es_retitulado",
    "es_adaptado",
    "cred_sup_tit",
    "cred_pend_sup_tit",
    "impagado_curso_mat",
]

MONTHLY_FAMILIES = [
    "pft_events",
    "pft_days_logged",
    "pft_visits",
    "pft_assignment_submissions",
    "pft_test_submissions",
    "pft_total_minutes",
    "n_wifi_days",
    "resource_events",
    "n_resource_days",
]
MONTHS = ("m09", "m10", "m11", "m12")
EXPECTED_NUMERIC_COUNT = 75
EXPECTED_CATEGORICAL_COUNT = 6
EXPECTED_FEATURE_COUNT = 81


@dataclass(frozen=True)
class Regime:
    name: str
    families: tuple
    delta_iqr: float
    cap_quantile: float


REGIMES = (
    Regime("STRICT", ("pft_days_logged", "pft_assignment_submissions", "pft_test_submissions", "n_resource_days"), 0.25, 0.95),
    Regime("MODERATE", ("pft_days_logged", "pft_assignment_submissions", "pft_test_submissions", "n_resource_days", "pft_visits", "pft_total_minutes", "resource_events"), 0.50, 0.975),
    Regime("FLEXIBLE", ("pft_days_logged", "pft_assignment_submissions", "pft_test_submissions", "n_resource_days", "pft_visits", "pft_total_minutes", "resource_events", "pft_events", "n_wifi_days"), 1.00, 0.99),
)

REFERENCE_SIZES = (10000, 50000, 100000, "full_train")
METHODS = ("UFCE-FF1", "UFCE-FF2", "UFCE-FF3", "DiCE", "AR")
TERMINAL_STATUSES = (
    "RUNTIME_ERROR",
    "TIMEOUT",
    "UNSUPPORTED",
    "NO_RETURNED_CF",
    "RETURNED_NON_FLIPPING_CF",
    "SUCCESS_VALID_CONSTRAINT_FAILED",
    "SUCCESS_VALID_FEASIBLE",
)

