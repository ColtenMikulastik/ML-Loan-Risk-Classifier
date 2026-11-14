
import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Load the German Credit dataset
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "statlog/german/german.data"
)

column_names = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings", "employment",
    "installment_rate", "personal_status", "other_debtors",
    "residence_since", "property", "age",
    "other_installment_plans", "housing",
    "existing_credits", "job", "dependents",
    "telephone", "foreign_worker", "credit_risk"
]

data = pd.read_csv(
    DATA_URL, sep=r"\s+", header=None, names=column_names
)

# Select 10 features for the application
features = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "age",
    "housing"
]

X = data[features]

# 1 = High Risk, 0 = Low Risk
y = (data["credit_risk"] == 2).astype(int)

# Split data into 80% training, 10% validation, 10% testing
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

X_validation, X_test, y_validation, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

categorical_features = [
    "checking_account",
    "credit_history",
    "purpose",
    "savings",
    "employment",
    "housing"
]

numerical_features = [
    "duration",
    "credit_amount",
    "installment_rate",
    "age"
]

# Prepare numerical and categorical features
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features),

    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# Create Logistic Regression and Decision Tree models
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),
    "Decision Tree": Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5, random_state=42
        ))
    ])
}

# Evaluate model performance
def evaluate_model(model, X_data, y_data):
    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)[:, 1]

    return {
        "Accuracy": accuracy_score(y_data, predictions),
        "Precision": precision_score(
            y_data, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_data, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_data, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_data, probabilities)
    }

# Create folder for trained models
Path("models").mkdir(exist_ok=True)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    print("\nValidation Results:")
    validation_results = evaluate_model(
        model, X_validation, y_validation
    )

    for metric, score in validation_results.items():
        print(f"{metric}: {score:.4f}")

    print("\nTest Results:")
    test_results = evaluate_model(model, X_test, y_test)

    for metric, score in test_results.items():
        print(f"{metric}: {score:.4f}")

    # Save trained model
    filename = (
        "logistic_regression.pkl"
        if model_name == "Logistic Regression"
        else "decision_tree.pkl"
    )

    joblib.dump(model, Path("models") / filename)

print("\nBoth models trained and saved successfully!")
