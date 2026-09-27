"""Reversible encoded search space; model evaluation always uses raw fields."""

import numpy as np
import pandas as pd


class EncodedSpace:
    def __init__(self, train_frame, schema):
        self.feature_names = list(schema["feature_names"])
        self.numeric = list(schema["numeric_features"])
        self.categorical = list(schema["categorical_features"])
        self.domains = {
            name: sorted(train_frame[name].astype(str).unique().tolist())
            for name in self.categorical
        }
        self.codes = {
            name: {value: code for code, value in enumerate(values)}
            for name, values in self.domains.items()
        }

    def encode(self, frame):
        result = pd.DataFrame(index=frame.index)
        for name in self.numeric:
            result[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        for name in self.categorical:
            values = frame[name].astype(str)
            encoded = []
            for value in values:
                if value not in self.codes[name]:
                    # Unseen categories are valid immutable query values. Give
                    # them a reversible representation instead of producing
                    # NaN, which breaks model preprocessing before the model's
                    # one-hot encoder can apply handle_unknown="ignore".
                    code = len(self.domains[name])
                    self.domains[name].append(value)
                    self.codes[name][value] = code
                encoded.append(self.codes[name][value])
            result[name] = pd.Series(encoded, index=frame.index, dtype=float)
        return result.loc[:, self.feature_names].reset_index(drop=True)

    def decode(self, frame):
        if not isinstance(frame, pd.DataFrame):
            frame = pd.DataFrame(frame, columns=self.feature_names)
        result = pd.DataFrame(index=frame.index)
        for name in self.numeric:
            result[name] = pd.to_numeric(frame[name], errors="coerce").astype(float)
        for name in self.categorical:
            reverse = {index: value for index, value in enumerate(self.domains[name])}
            codes = pd.to_numeric(frame[name], errors="coerce")
            decoded = []
            for value in codes:
                if pd.isna(value) or not float(value).is_integer():
                    # Make malformed categorical proposals an explicit unknown
                    # category. The frozen model can score it as all-zero OHE,
                    # then the verifier rejects the immutable-field change.
                    decoded.append("__UFCE_INVALID_CATEGORY__")
                else:
                    decoded.append(reverse.get(int(value), "__UFCE_INVALID_CATEGORY__"))
            result[name] = pd.Series(decoded, index=frame.index, dtype="string")
        return result.loc[:, self.feature_names]

    def row_key(self, row):
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return tuple(row[name] for name in self.feature_names)
