"""Three views of "which features matter", all reported per ORIGINAL attribute
where possible (so 'checking_status' is one bar, not four one-hot columns).

1. logistic_coefficients      - sign and size of each encoded column's weight
2. tree_importance_by_feature - how much each attribute reduced the tree's impurity
3. permutation_by_feature     - how much accuracy drops when one attribute is shuffled

Importance describes what the MODEL uses, not what causes default, and
correlated features can share or hide importance.
"""

import pandas as pd
from sklearn.inspection import permutation_importance

from src.train import RANDOM_STATE


def attribute_of(encoded_name: str, features: list[str]) -> str:
    """Map a ColumnTransformer output name back to its original attribute.

    'nominal__checking_status_<0 DM' -> 'checking_status'
    'numeric__age'                   -> 'age'
    """
    rest = encoded_name.split("__", 1)[1]
    matches = [f for f in features if rest == f or rest.startswith(f + "_")]
    return max(matches, key=len)  # longest match, e.g. other_debtors vs other_...


def readable_name(encoded_name: str, features: list[str]) -> str:
    """'nominal__checking_status_<0 DM' -> 'checking_status = <0 DM'."""
    attr = attribute_of(encoded_name, features)
    rest = encoded_name.split("__", 1)[1]
    return attr if rest == attr else f"{attr} = {rest[len(attr) + 1:]}"


def logistic_coefficients(model, features: list[str]) -> pd.Series:
    """Coefficient per encoded column, sorted by absolute size (largest first).

    Positive = pushes toward bad credit, negative = toward good credit.
    Numeric/ordinal columns are scaled, so their sizes are comparable.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    coefs = model.named_steps["clf"].coef_[0]
    s = pd.Series(coefs, index=[readable_name(n, features) for n in names])
    return s.reindex(s.abs().sort_values(ascending=False).index)


def tree_importance_by_feature(model, features: list[str]) -> pd.Series:
    """Tree importance summed over each attribute's one-hot columns, largest first."""
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["clf"].feature_importances_, index=names)
    by_attr = imp.groupby(lambda n: attribute_of(n, features)).sum()
    return by_attr.sort_values(ascending=False)


def permutation_by_feature(model, X, y, features: list[str], n_repeats: int = 20) -> pd.DataFrame:
    """Mean/std drop in accuracy when each ORIGINAL column is shuffled.

    Works on any fitted Pipeline, because shuffling happens before encoding.
    Pass validation data (never test). Columns: mean, std; sorted by mean.
    """
    r = permutation_importance(
        model, X[list(features)], y,
        scoring="accuracy", n_repeats=n_repeats, random_state=RANDOM_STATE,
    )
    out = pd.DataFrame({"mean": r.importances_mean, "std": r.importances_std}, index=list(features))
    return out.sort_values("mean", ascending=False)
