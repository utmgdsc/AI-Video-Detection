import os
import pickle
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    input_csv = "stacking_train_val.csv"
    output_model_path = "./backend/models/stacking/stacking_model.pkl"

    df = pd.read_csv(input_csv)

    feature_cols = [
        "aasist_score",
        "efficientnet_score",
        "mesonet_score",
        "xceptionnet_score",
    ]

    # convert to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows where all model scores are missing
    df = df.dropna(subset=feature_cols, how="all")

    # fill missing scores with 0.5 for now
    df[feature_cols] = df[feature_cols].fillna(0.5)

    # optional: remove rows with too many missing values
    missing_count = (df[feature_cols] == 0.5).sum(axis=1)
    df = df[missing_count <= 2].copy()

    X = df[feature_cols]
    y = df["label"]

    print("Label counts:")
    print(y.value_counts())

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_val)

    print("Validation Accuracy:", accuracy_score(y_val, preds))
    print(classification_report(y_val, preds))

    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)

    with open(output_model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved stacking model to {output_model_path}")


if __name__ == "__main__":
    main()