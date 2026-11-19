"""Streamlit app: pick features and hyperparameters, retrain live, predict, compare.

Run from the repo root:  uv run streamlit run app.py

Tabs: predict one applicant / model performance / feature importance / final test.
The sidebar controls which features and hyperparameters the models use; every
change retrains both models (it takes milliseconds on 1,000 rows).
"""

import pandas as pd
import streamlit as st

from src.features import ALL_FEATURES
from src.importance import (
    logistic_coefficients,
    permutation_by_feature,
    tree_importance_by_feature,
)
from src.parse import SCHEMA
from src.train import (
    DEFAULT_C,
    DEFAULT_MAX_DEPTH,
    baseline_model,
    build_model,
    evaluate,
    split_data,
)

st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="wide")

# Slider options. None = unlimited tree depth.
C_OPTIONS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
DEPTH_OPTIONS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

TOP_N = 12  # keep the importance charts readable
RAISES, LOWERS = "raises risk", "lowers risk"
RAISES_COLOR, LOWERS_COLOR = "#D55E00", "#0072B2"  # two colour-blind-safe hues, not red/green


# ---- Cached helpers ---------------------------------------------------------
# Streamlit re-runs this whole script on every widget click. Caching means the
# data is split once, and each (model, features, hyperparameter) combination is
# trained once; going back to a combination you already tried is instant.
@st.cache_data
def get_split():
    return split_data()


@st.cache_resource
def fit_model(name: str, features: tuple[str, ...], C: float, max_depth: int | None):
    """Train on the TRAINING split only. features is a tuple so it can be cache-keyed."""
    X_train, _, _, y_train, _, _ = get_split()
    return build_model(name, list(features), C=C, max_depth=max_depth).fit(X_train, y_train)


@st.cache_resource
def fit_baseline():
    X_train, _, _, y_train, _, _ = get_split()
    return baseline_model().fit(X_train, y_train)


@st.cache_data
def get_permutation(name: str, features: tuple[str, ...], C: float, max_depth: int | None):
    """Permutation importance on the VALIDATION set (cached per configuration)."""
    _, X_val, _, _, y_val, _ = get_split()
    model = fit_model(name, features, C, max_depth)
    return permutation_by_feature(model, X_val, y_val, list(features))


def confusion_frame(cm) -> pd.DataFrame:
    """Rows = actual class, columns = predicted class (bad credit is the positive class)."""
    return pd.DataFrame(
        cm,
        index=["actual good", "actual bad"],
        columns=["predicted good", "predicted bad"],
    )


def feature_label(name: str) -> str:
    return name.replace("_", " ").capitalize()


# ---- Sidebar: features and hyperparameters ---------------------------------
st.sidebar.header("Features")
st.sidebar.caption(
    "Remove or add features. Both models are retrained on the chosen features "
    "every time you change this."
)
features = st.sidebar.multiselect(
    "Features used by the models",
    options=ALL_FEATURES,
    default=ALL_FEATURES,
    format_func=lambda f: f"{f}  ({SCHEMA[f][0]})",
)

st.sidebar.header("Hyperparameters")
C = st.sidebar.select_slider(
    "Logistic regression C (smaller = simpler model)",
    options=C_OPTIONS,
    value=DEFAULT_C,
)
max_depth = st.sidebar.select_slider(
    "Decision tree max depth",
    options=DEPTH_OPTIONS,
    value=DEFAULT_MAX_DEPTH,
    format_func=lambda d: "unlimited" if d is None else str(d),
)

# ---- Main page --------------------------------------------------------------
st.title("Credit Risk Prediction")
st.write(
    "Logistic regression vs. decision tree on the Statlog German Credit data. "
    "**Bad credit is the positive class** (the applicant is predicted to default)."
)

if not features:
    st.warning("Select at least one feature in the sidebar.")
    st.stop()

X_train, X_val, X_test, y_train, y_val, y_test = get_split()
feature_key = tuple(features)  # tuple so Streamlit can cache on it

models = {
    "Majority baseline (always 'good')": fit_baseline(),
    "Logistic regression": fit_model("logistic", feature_key, C, max_depth),
    "Decision tree": fit_model("tree", feature_key, C, max_depth),
}

tab_predict, tab_perf, tab_imp, tab_test = st.tabs([
    "Predict an applicant",
    "Model performance",
    "Which features matter?",
    "Final test",
])

# ---- Tab 1: predict one applicant ------------------------------------------
with tab_predict:
    st.caption(
        "Describe an applicant and both models predict whether they are a good or bad "
        "credit risk. The form only asks about the features selected in the sidebar. "
        "It starts on a typical applicant: the most common category, and the median "
        "number, in the training data."
    )

    applicant = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        kind, codes = SCHEMA[f]
        col = cols[i % 3]
        label = feature_label(f)
        if kind == "numeric":
            lo, hi = int(X_train[f].min()), int(X_train[f].max())
            applicant[f] = col.number_input(
                f"{label}  ({lo} to {hi} in the data)",
                min_value=lo, max_value=hi, value=int(X_train[f].median()),
                key=f"input_{f}",
            )
        else:  # nominal or ordinal: pick from the readable labels in SCHEMA order
            options = list(codes.values())
            typical = X_train[f].mode()[0]
            applicant[f] = col.selectbox(
                label, options, index=options.index(typical), key=f"input_{f}",
            )

    row = pd.DataFrame([applicant])
    st.subheader("Predictions")
    result_cols = st.columns(2)
    for col, (name, key) in zip(
        result_cols, (("Logistic regression", "logistic"), ("Decision tree", "tree"))
    ):
        model = models[name]
        predicted_bad = model.predict(row)[0] == 1
        prob_bad = model.predict_proba(row)[0][1]
        col.markdown(f"**{name}**")
        if predicted_bad:
            col.error("Predicted: HIGH credit risk (bad credit)")
        else:
            col.success("Predicted: LOW credit risk (good credit)")
        col.metric("Estimated probability of bad credit", f"{prob_bad * 100:.1f}%")

    st.caption(
        "A class is predicted 'bad' when its probability is above 50%. A shallow tree "
        "gives coarse probabilities (each leaf is just the share of bad loans in it). "
        "This is an educational model trained on historical German credit data, not a "
        "lending decision."
    )

# ---- Tab 2: model performance ----------------------------------------------
with tab_perf:
    st.caption(
        f"How well do the models do with the {len(features)} selected features? "
        f"Trained on {len(X_train)} rows, checked on {len(X_val)} validation rows they "
        "have not seen. A big gap between train and validation accuracy means "
        "overfitting. The baseline always says 'good', so any useful model should beat it."
    )

    rows, val_results = [], {}
    for name, model in models.items():
        train_r = evaluate(model, X_train, y_train)
        val_r = evaluate(model, X_val, y_val)
        val_results[name] = val_r
        rows.append({
            "Model": name,
            "Train accuracy": train_r["accuracy"],
            "Validation accuracy": val_r["accuracy"],
            "Precision": val_r["precision"],
            "Recall": val_r["recall"],
            "F1": val_r["f1"],
        })
    st.dataframe(
        pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"),
        width="stretch",
    )
    st.caption(
        "Precision, recall and F1 are on the validation set. Precision: of the people "
        "flagged bad, how many really were. Recall: of the people who really were bad, "
        "how many were caught."
    )

    st.subheader("Confusion matrices (validation set)")
    st.caption("Rows are the true class, columns what the model predicted.")
    cols = st.columns(len(models))
    for col, (name, r) in zip(cols, val_results.items()):
        col.markdown(f"**{name}**")
        col.dataframe(confusion_frame(r["confusion"]))

# ---- Tab 3: feature importance ----------------------------------------------
with tab_imp:
    st.caption(
        "Which attributes do the models rely on? Importance describes what each model "
        "*uses*, not what causes default, and correlated features can share or hide "
        "importance. Change the features in the sidebar and watch these shift."
    )

    perm_section, internal_section = st.tabs(["Permutation importance", "Inside each model"])

    with perm_section:
        st.caption(
            "Shuffle one attribute's values in the validation set and measure how much "
            "accuracy drops. A bigger drop means the model relied on it more. It works "
            f"the same way for both models. Validation has only {len(y_val)} rows, so "
            "small differences are noise; the table shows the spread."
        )
        left, right = st.columns(2)
        for col, (label, key) in zip(
            (left, right), (("Logistic regression", "logistic"), ("Decision tree", "tree"))
        ):
            perm = get_permutation(key, feature_key, C, max_depth)
            col.markdown(f"**{label}**")
            chart = perm.head(TOP_N).rename_axis("feature").reset_index()
            col.bar_chart(
                chart, x="feature", y="mean", horizontal=True, sort=False,
                x_label="", y_label="accuracy drop when shuffled",
            )
            with col.expander("Show values (mean ± std over 20 shuffles)"):
                st.dataframe(perm.style.format("{:.3f}"))

    with internal_section:
        left, right = st.columns(2)

        left.markdown("**Logistic regression: coefficients**")
        left.caption(
            "Weight per encoded column on the scaled features. Orange pushes toward bad "
            "credit, blue toward good credit."
        )
        coefs = logistic_coefficients(models["Logistic regression"], features).head(TOP_N)
        coef_chart = pd.DataFrame({
            "feature": coefs.index,
            RAISES: coefs.clip(lower=0).values,
            LOWERS: coefs.clip(upper=0).values,
        })
        left.bar_chart(
            coef_chart, x="feature", y=[RAISES, LOWERS], color=[RAISES_COLOR, LOWERS_COLOR],
            horizontal=True, sort=False, stack=True, x_label="", y_label="coefficient",
        )

        right.markdown("**Decision tree: importance**")
        right.caption(
            "Share of the tree's impurity reduction that each attribute is responsible "
            "for (adds up to 100%). Similar to MATLAB's predictor importance."
        )
        tree_imp = tree_importance_by_feature(models["Decision tree"], features).head(TOP_N)
        tree_chart = tree_imp.rename("importance").rename_axis("feature").reset_index()
        right.bar_chart(
            tree_chart, x="feature", y="importance", horizontal=True, sort=False,
            x_label="", y_label="share of total importance",
        )

# ---- Tab 4: final test-set evaluation --------------------------------------
with tab_test:
    st.caption(
        "The test set should be looked at once, after you've settled on features and "
        "hyperparameters. If you keep adjusting the sidebar in response to these numbers, "
        "you are tuning on the test set and the results stop being honest."
    )
    if st.button("Evaluate on test set (final)"):
        test_rows = []
        test_cm = {}
        for name, model in models.items():
            r = evaluate(model, X_test, y_test)
            test_cm[name] = r["confusion"]
            test_rows.append({
                "Model": name,
                "Accuracy": r["accuracy"],
                "Precision": r["precision"],
                "Recall": r["recall"],
                "F1": r["f1"],
            })
        st.dataframe(
            pd.DataFrame(test_rows).set_index("Model").style.format("{:.3f}"),
            width="stretch",
        )
        cols = st.columns(len(models))
        for col, (name, cm) in zip(cols, test_cm.items()):
            col.markdown(f"**{name}**")
            col.dataframe(confusion_frame(cm))
