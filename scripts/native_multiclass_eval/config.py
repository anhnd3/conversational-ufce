"""Frozen design choices for the student-outcome multiclass experiment."""

from dataclasses import dataclass


DATASET_ID = 697
DATASET_URL = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
EXPECTED_ROWS = 4424
EXPECTED_FEATURES = 36
TARGET = "Target"
CLASS_NAMES = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}
CLASS_LABELS = {value: key for key, value in CLASS_NAMES.items()}
SPLIT_SEED = 42
PILOT_SEED = 42
BOOTSTRAP_SEED = 42
BOOTSTRAP_RESAMPLES = 2000
MAX_CANDIDATES = 5
DICE_SAMPLE_SIZE = 1000
TIMEOUT_SECONDS = 60.0
CHECKPOINT_INTERVAL_QUERIES = 10
MI_TOP_K = 10
UFCE_RADIUS = 1e9
UFCE_NEIGHBORS = 1000

CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Application order",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ACTIONABLE_FEATURES = [
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

INVARIANT_CANDIDATES = (
    ("approved_le_enrolled", "approved", "le", "enrolled", "Approved curricular units are a subset of enrolled units."),
    ("approved_le_evaluations", "approved", "le", "evaluations", "Each unit counted as approved should have at least one evaluation."),
    ("without_eval_le_enrolled", "without_evaluations", "le", "enrolled", "Units without evaluations are a subset of enrolled units."),
    ("approved_plus_without_eval_le_enrolled", "approved_plus_without_evaluations", "le", "enrolled", "Approved units and units without evaluations are disjoint statuses among enrolled units."),
)


@dataclass(frozen=True)
class Transition:
    source: int
    target: int

    @property
    def key(self):
        return "%s_to_%s" % (CLASS_NAMES[self.source].lower(), CLASS_NAMES[self.target].lower())


TRANSITIONS = tuple(
    Transition(source, target)
    for source in range(3)
    for target in range(3)
    if source != target
)
PILOT_COUNTS = {
    "dropout_to_enrolled": 4,
    "dropout_to_graduate": 4,
    "enrolled_to_dropout": 3,
    "enrolled_to_graduate": 3,
    "graduate_to_dropout": 3,
    "graduate_to_enrolled": 3,
}
METHODS = ("UFCE-FF1", "UFCE-FF2", "UFCE-FF3", "DiCE")
