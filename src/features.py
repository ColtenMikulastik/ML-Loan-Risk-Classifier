"""Build the preprocessing step for any subset of the 20 features.

We use a scikit-learn ColumnTransformer to help reduce data leakage: the
scaler's mean/std are learned from the TRAINING rows only (when .fit() is
called on the training split) and then merely applied to validation/test rows.
SCHEMA in parse.py is the single source of truth for each feature's type.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.parse import SCHEMA

ALL_FEATURES = list(SCHEMA)  # the 20 attributes, in german.doc order


def build_preprocessor(features: list[str]) -> ColumnTransformer:
    """Return an (unfitted) ColumnTransformer for the chosen features.

    numeric -> StandardScaler            (logistic regression needs similar scales; trees ignore it)
    nominal -> OneHotEncoder             (unordered categories become 0/1 columns)
    ordinal -> OrdinalEncoder + scaling  (keeps the order: unemployed < <1 yr < ...)
    """
    def of_kind(kind):
        return [f for f in features if SCHEMA[f][0] == kind]

    def levels(cols):  # category lists come from SCHEMA, not from the data
        return [list(SCHEMA[c][1].values()) for c in cols]

    numeric, nominal, ordinal = of_kind("numeric"), of_kind("nominal"), of_kind("ordinal")

    steps = []
    if numeric:
        steps.append(("numeric", StandardScaler(), numeric))
    if nominal:
        steps.append(("nominal", OneHotEncoder(categories=levels(nominal)), nominal))
    if ordinal:
        steps.append(("ordinal", Pipeline([
            ("encode", OrdinalEncoder(categories=levels(ordinal))),
            ("scale", StandardScaler()),
        ]), ordinal))

    return ColumnTransformer(steps)
