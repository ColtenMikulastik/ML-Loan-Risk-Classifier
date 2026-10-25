"""Parse the Statlog German Credit data into analysis-ready CSVs.

Two stages, kept separate on purpose so each can be inspected and plotted:

1. ``load_clean``    -> readable pandoc DataFrame (decoded labels, proper dtypes). Use for EDA / figures.
2. ``encode_features`` -> numeric pandoc DataFrame (one-hot + ordinal ints). Use for tree / logistic regression.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Column order matches german.doc attributes 1-20, then the label.
# Each entry: name -> (kind, codes). kind is "numeric", "nominal" or "ordinal".
# numeric: value
# nominal: non-ordered classifications (must be broken down into one-hot encodings)
# ordinal: ordered classifications
# For categorical kinds, codes maps raw code -> readable label; for "ordinal",
# which is then encoded starting at 0
# dict order is the rank order (lowest first).
SCHEMA: dict[str, tuple[str, dict[str, str] | None]] = {
    "checking_status": ("nominal", {
        "A11": "<0 DM", "A12": "0-200 DM", "A13": ">=200 DM", "A14": "no account"}),
    "duration_months": ("numeric", None),
    "credit_history": ("nominal", {
        "A30": "no credits/all paid", "A31": "all paid at this bank",
        "A32": "existing paid duly", "A33": "past delay", "A34": "critical/other credits"}),
    "purpose": ("nominal", {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture", "A43": "radio/tv",
        "A44": "appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "other"}),
    "credit_amount": ("numeric", None),
    "savings": ("nominal", {
        "A61": "<100 DM", "A62": "100-500 DM", "A63": "500-1000 DM",
        "A64": ">=1000 DM", "A65": "unknown/none"}),
    "employment_since": ("ordinal", {
        "A71": "unemployed", "A72": "<1 yr", "A73": "1-4 yrs", "A74": "4-7 yrs", "A75": ">=7 yrs"}),
    "installment_rate": ("numeric", None),
    "personal_status_sex": ("nominal", {
        "A91": "male div/sep", "A92": "female div/sep/married", "A93": "male single",
        "A94": "male married/widowed", "A95": "female single"}),
    "other_debtors": ("nominal", {"A101": "none", "A102": "co-applicant", "A103": "guarantor"}),
    "residence_since": ("numeric", None),
    "property": ("nominal", {
        "A121": "real estate", "A122": "savings agmt/insurance",
        "A123": "car/other", "A124": "unknown/none"}),
    "age": ("numeric", None),
    "other_installment_plans": ("nominal", {"A141": "bank", "A142": "stores", "A143": "none"}),
    "housing": ("nominal", {"A151": "rent", "A152": "own", "A153": "free"}),
    "existing_credits": ("numeric", None),
    "job": ("ordinal", {
        "A171": "unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled", "A174": "management/highly qualified"}),
    "dependents": ("numeric", None),
    "telephone": ("nominal", {"A191": "none", "A192": "yes"}),
    "foreign_worker": ("nominal", {"A201": "yes", "A202": "no"}),
}
TARGET = "bad_credit"  # 1 = bad (the costly class to miss, see cost matrix in german.doc)


def load_clean(path: Path = DATA_DIR / "german.data") -> pd.DataFrame:
    """Read german.data into a typed, human-readable DataFrame."""
    names = [*SCHEMA, "raw_label"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype=str)

    # check for errors
    if len(df) != 1000:
        raise ValueError(f"expected 1000 rows, got {len(df)}")
    if df.isna().any().any():
        raise ValueError("missing values found; file may be malformed")

    # use Schema to figure out encoding, depending on type
    for col, (kind, codes) in SCHEMA.items():
        if kind == "numeric":
            df[col] = pd.to_numeric(df[col], errors="raise")
            continue
        unknown = set(df[col]) - set(codes)
        if unknown:  # fail loudly rather than silently producing NaN
            raise ValueError(f"{col}: unexpected codes {sorted(unknown)}")
        df[col] = pd.Categorical(
            df[col].map(codes), categories=list(codes.values()), ordered=(kind == "ordinal")
        )

    # Raw label: 1 = good, 2 = bad. Recode so "positive" = the risky class.
    if not set(df["raw_label"]) <= {"1", "2"}:
        raise ValueError(f"unexpected labels {set(df['raw_label'])}")
    df[TARGET] = (df.pop("raw_label") == "2").astype(int)
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Model-ready numeric frame: ordinals -> rank ints, nominals -> one-hot.

    Deliberately NOT scaled here: scaling must be fit on the training split only
    (otherwise test-set statistics leak into training). Do it in the modelling step.
    """
    out = df.copy()
    nominal = [c for c, (k, _) in SCHEMA.items() if k == "nominal"]
    for col, (kind, _) in SCHEMA.items():
        if kind == "ordinal":
            out[col] = out[col].cat.codes
    # Keep every dummy (no drop_first): trees don't care, and for logistic regression
    # use regularisation (sklearn default) or drop one level per feature yourself.
    out = pd.get_dummies(out, columns=nominal, prefix_sep="=", dtype=int)
    return out


def main() -> None:
    """ clean, decode, then encode data into csv """
    clean = load_clean()
    encoded = encode_features(clean)
    clean.to_csv(DATA_DIR / "german_clean.csv", index=False)
    encoded.to_csv(DATA_DIR / "german_encoded.csv", index=False)
    print(f"clean:   {clean.shape}  -> data/german_clean.csv")
    print(f"encoded: {encoded.shape}  -> data/german_encoded.csv")
    print(f"bad-credit rate: {clean[TARGET].mean():.1%}")


if __name__ == "__main__":
    main()
