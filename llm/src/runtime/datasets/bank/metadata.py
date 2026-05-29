from __future__ import annotations


BANK_SCHEMA_VERSION = "bank_schema_v1"
BANK_POLICY_VERSION = "bank_policy_v1"
BANK_PARSER_SCHEMA_VERSION = "parser_schema_v3"
BANK_CURRENCY_CONVERSION_MODE = "allow_usd_and_vnd_heuristic"
BANK_VND_USD_FX_RATE = 25000
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
    "Income": [
        "income",
        "annual income",
        "yearly income",
        "salary",
        "earn",
        "earning",
        "make",
        "thu nhap",
        "thu nhập",
        "luong",
        "lương",
        "muc luong",
        "mức lương",
        "luong nam",
        "lương năm",
        "thu nhap nam",
        "thu nhập năm",
    ],
    "Family": [
        "family",
        "family size",
        "household",
        "household size",
        "gia dinh",
        "gia đình",
        "so nguoi trong gia dinh",
        "số người trong gia đình",
        "ho gia dinh",
        "hộ gia đình",
        "quy mo ho",
        "quy mô hộ",
    ],
    "CCAvg": [
        "ccavg",
        "cc avg",
        "credit card avg",
        "average cc spend",
        "average card spending",
        "credit card spending",
        "card spending",
        "card average",
        "card average spending",
        "monthly card spending",
        "spend",
        "spending",
        "chi tieu the",
        "chi tiêu thẻ",
        "chi tieu the tin dung",
        "chi tiêu thẻ tín dụng",
        "muc chi tieu the",
        "mức chi tiêu thẻ",
        "chi tieu hang thang",
        "chi tiêu hàng tháng",
        "trung binh the",
        "trung bình thẻ",
    ],
    "Education": [
        "education",
        "education code",
        "education level",
        "education category",
        "hoc van",
        "học vấn",
        "trinh do hoc van",
        "trình độ học vấn",
        "trinh do hoc",
        "trình độ học",
    ],
    "Mortgage": [
        "mortgage",
        "the chap",
        "thế chấp",
        "khoan the chap",
        "khoản thế chấp",
        "vay the chap",
        "vay thế chấp",
    ],
    "SecuritiesAccount": [
        "securities account",
        "securitiesaccount",
        "security account",
        "investment account",
        "tai khoan chung khoan",
        "tài khoản chứng khoán",
        "tk chung khoan",
        "tk chứng khoán",
    ],
    "CDAccount": [
        "cd account",
        "cdaccount",
        "certificate of deposit",
        "tai khoan tien gui co ky han",
        "tài khoản tiền gửi có kỳ hạn",
        "so tiet kiem co ky han",
        "sổ tiết kiệm có kỳ hạn",
    ],
    "Online": [
        "online",
        "online banking",
        "ngan hang truc tuyen",
        "ngân hàng trực tuyến",
        "internet banking",
    ],
    "CreditCard": [
        "credit card",
        "creditcard",
        "bank credit card",
        "the tin dung",
        "thẻ tín dụng",
    ],
}

# v3 bilingual hard aliases in canonical Unicode (kept outside the literal map to avoid mojibake drift).
_BANK_V3_UNICODE_ALIASES = {
    "Income": [
        "thu nh\u1eadp",
        "l\u01b0\u01a1ng",
        "m\u1ee9c l\u01b0\u01a1ng",
        "l\u01b0\u01a1ng n\u0103m",
        "thu nh\u1eadp n\u0103m",
    ],
    "Family": [
        "gia \u0111\u00ecnh",
        "s\u1ed1 ng\u01b0\u1eddi trong gia \u0111\u00ecnh",
        "h\u1ed9 gia \u0111\u00ecnh",
        "quy m\u00f4 h\u1ed9",
    ],
    "CCAvg": [
        "chi ti\u00eau th\u1ebb",
        "chi ti\u00eau th\u1ebb t\u00edn d\u1ee5ng",
        "m\u1ee9c chi ti\u00eau th\u1ebb",
        "chi ti\u00eau h\u00e0ng th\u00e1ng",
        "trung b\u00ecnh th\u1ebb",
    ],
    "Education": [
        "h\u1ecdc v\u1ea5n",
        "tr\u00ecnh \u0111\u1ed9 h\u1ecdc v\u1ea5n",
        "tr\u00ecnh \u0111\u1ed9 h\u1ecdc",
    ],
    "Mortgage": [
        "th\u1ebf ch\u1ea5p",
        "kho\u1ea3n th\u1ebf ch\u1ea5p",
        "vay th\u1ebf ch\u1ea5p",
    ],
    "SecuritiesAccount": [
        "t\u00e0i kho\u1ea3n ch\u1ee9ng kho\u00e1n",
        "tk ch\u1ee9ng kho\u00e1n",
    ],
    "CDAccount": [
        "t\u00e0i kho\u1ea3n ti\u1ec1n g\u1eedi c\u00f3 k\u1ef3 h\u1ea1n",
        "s\u1ed5 ti\u1ebft ki\u1ec7m c\u00f3 k\u1ef3 h\u1ea1n",
    ],
    "Online": [
        "ng\u00e2n h\u00e0ng tr\u1ef1c tuy\u1ebfn",
    ],
    "CreditCard": [
        "th\u1ebb t\u00edn d\u1ee5ng",
    ],
}
for _field_name, _extra_aliases in _BANK_V3_UNICODE_ALIASES.items():
    BANK_ALIASES.setdefault(_field_name, [])
    for _alias in _extra_aliases:
        if _alias not in BANK_ALIASES[_field_name]:
            BANK_ALIASES[_field_name].append(_alias)

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
