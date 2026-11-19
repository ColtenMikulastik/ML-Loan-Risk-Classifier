"""Split the data, build the two models, and evaluate them.

Run the cross-validated grid search (never touches the test set):
    uv run python -m src.train
Evaluate the chosen configuration on the test set (do this once, at the end):
    uv run python -m src.train --test
"""

import sys

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.features import ALL_FEATURES, build_preprocessor
from src.parse import TARGET, load_clean

# ---- Named constants -------------------------------------------------------
RANDOM_STATE = 42  # fixed seed so every run (and teammate) gets the same split

# Default hyperparameters = the values the 5-fold CV grid search picked on the
# full 20-feature set. The UI sliders start here.
DEFAULT_C = 0.1          # logistic regression: inverse regularisation strength (smaller = simpler)
DEFAULT_MAX_DEPTH = 6    # tree: limits depth to reduce overfitting (None = unlimited)

# Grids searched on the validation set, ordered simplest -> most complex so a
# tie in validation accuracy is won by the simpler setting.
C_GRID = [0.01, 0.1, 1, 10, 100]
DEPTH_GRID = [1, 2, 3, 4, 5, 6, 8, 10, None]  # None = unlimited depth


def split_data(random_state: int = RANDOM_STATE):
    """Stratified 80/10/10 train/validation/test split, shared by both models.

    Stratified so each part keeps the ~30% bad-credit rate. Returns
    X_train, X_val, X_test, y_train, y_val, y_test.
    """
    df = load_clean()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(name: str, features: list[str] = ALL_FEATURES,
                C: float = DEFAULT_C, max_depth: int | None = DEFAULT_MAX_DEPTH) -> Pipeline:
    """Unfitted Pipeline(preprocessing -> classifier) for "logistic" or "tree".

    Putting preprocessing inside the Pipeline means .fit() learns the scaler on
    the training rows only (helps reduce leakage). The ColumnTransformer selects
    `features` by name, so the full DataFrame can be passed in.
    """
    if name == "logistic":
        clf = LogisticRegression(C=C, max_iter=1000)
    elif name == "tree":
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"unknown model {name!r}")
    return Pipeline([("prep", build_preprocessor(features)), ("clf", clf)])


def baseline_model() -> DummyClassifier:
    """Majority-class baseline: predicts 'good credit' for everyone."""
    return DummyClassifier(strategy="most_frequent")


def evaluate(model, X, y) -> dict:
    """Metrics with bad credit (1) as the positive class.

    confusion is [[TN, FP], [FN, TP]]: rows = actual, columns = predicted.
    """
    pred = model.predict(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "confusion": confusion_matrix(y, pred, labels=[0, 1]),
    }


N_FOLDS = 5  # cross-validation folds used for tuning (on the training set only)


def grid_search(name, grid, param, split):
    """Choose a hyperparameter by 5-fold cross-validation on the TRAINING set.

    With only ~100 validation rows, one applicant moves accuracy by a full
    point, so picking from validation alone is noisy. Cross-validation scores
    each setting on 5 different held-out folds of the 800 training rows.
    cross_val_score clones the whole Pipeline per fold, so the scaler/encoders
    are re-fitted on each fold's training part only (no leakage).

    The validation set is NOT used to choose; it is printed as an untouched check.
    Returns the value with the best mean CV accuracy (ties -> the simpler value).
    """
    X_train, X_val, _, y_train, y_val, _ = split
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    best_value, best_cv = None, -1.0
    print(f"\n{name}: {param}   train acc   CV acc   val acc (check only)")
    for value in grid:
        model = build_model(name, **{param: value})
        cv_acc = cross_val_score(model, X_train, y_train, cv=folds, scoring="accuracy").mean()
        model.fit(X_train, y_train)
        train_acc = evaluate(model, X_train, y_train)["accuracy"]
        val_acc = evaluate(model, X_val, y_val)["accuracy"]
        print(f"  {value!s:>8}        {train_acc:.3f}     {cv_acc:.3f}     {val_acc:.3f}")
        if cv_acc > best_cv:  # strict ">" keeps the earlier (simpler) value on ties
            best_value, best_cv = value, cv_acc
    print(f"  -> chosen {param} = {best_value} (CV acc {best_cv:.3f})")
    return best_value


def main():
    split = split_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split
    print(f"train {len(X_train)}  val {len(X_val)}  test {len(X_test)}")

    base = baseline_model().fit(X_train, y_train)
    print(f"majority baseline: val acc {evaluate(base, X_val, y_val)['accuracy']:.3f}")

    best_C = grid_search("logistic", C_GRID, "C", split)
    best_depth = grid_search("tree", DEPTH_GRID, "max_depth", split)

    if "--test" in sys.argv:  # final, one-time look at the test set
        print("\n=== TEST SET (final) ===")
        models = {
            "baseline": base,
            "logistic": build_model("logistic", C=best_C).fit(X_train, y_train),
            "tree": build_model("tree", max_depth=best_depth).fit(X_train, y_train),
        }
        for name, model in models.items():
            r = evaluate(model, X_test, y_test)
            print(f"\n{name}: acc {r['accuracy']:.3f}  prec {r['precision']:.3f}  "
                  f"rec {r['recall']:.3f}  f1 {r['f1']:.3f}\n{r['confusion']}")


if __name__ == "__main__":
    main()
