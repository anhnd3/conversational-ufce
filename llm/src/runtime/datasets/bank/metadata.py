from __future__ import annotations


BANK_SCHEMA_VERSION = "bank_schema_v1"
BANK_POLICY_VERSION = "bank_policy_v1"
BANK_STEP = {
    "Income": 1,
    "CCAvg": 0.1,
    "Family": 1,
    "Education": 1,
    "Mortgage": 1,
    "CDAccount": 1,
    "Online": 1,
    "SecuritiesAccount": 1,
    "CreditCard": 1,
}
BANK_STEP_PROVENANCE = (
    "Copied from legacy UFCE bank source assumptions and behavior in ufce/core/data_processing.py; "
    "not derived from user input and not learned online."
)
BANK_ALIASES = {
    "Income": ["income", "salary"],
    "Family": ["family", "family size", "household"],
    "CCAvg": [
        "ccavg",
        "cc avg",
        "credit card avg",
        "average cc spend",
        "credit card spending",
        "card spending",
    ],
    "Education": ["education"],
    "Mortgage": ["mortgage"],
    "SecuritiesAccount": ["securities account", "securitiesaccount", "security account", "investment account"],
    "CDAccount": ["cd account", "cdaccount", "certificate of deposit"],
    "Online": ["online", "online banking"],
    "CreditCard": ["credit card", "creditcard", "bank credit card"],
}
BANK_FEATURE_TYPES = {
    "Income": "float",
    "Family": "int",
    "CCAvg": "float",
    "Education": "int",
    "Mortgage": "float",
    "SecuritiesAccount": "binary",
    "CDAccount": "binary",
    "Online": "binary",
    "CreditCard": "binary",
}
BANK_REQUIRED_FIELD_ORDER = tuple(BANK_FEATURE_TYPES.keys())
BANK_BOOLEAN_FIELDS = tuple(
    field_name for field_name, feature_type in BANK_FEATURE_TYPES.items() if feature_type == "binary"
)
BANK_FROZEN_MI_FEATURE_PAIRS = [
    ["CCAvg", "Income"],
    ["CDAccount", "CCAvg"],
    ["CDAccount", "Income"],
    ["Mortgage", "CCAvg"],
    ["CDAccount", "Mortgage"],
]
