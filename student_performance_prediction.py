"""Predict student final scores with Linear Regression and Random Forest."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "Student_Performance.csv"
TARGET = "Final_Score"
RANDOM_STATE = 42


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in the dataset.")

    print("Dataset preview:\n", df.head())
    print("\nDataset information:")
    df.info()
    print("\nMissing values:\n", df.isnull().sum())

    X = pd.get_dummies(df.drop(columns=[TARGET]), drop_first=False)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
    }

    results = []
    predictions = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, pred),
            "MSE": mean_squared_error(y_test, pred),
            "R2": r2_score(y_test, pred),
        })
        print(f"\n{name}: MAE={results[-1]['MAE']:.3f}, "
              f"MSE={results[-1]['MSE']:.3f}, R2={results[-1]['R2']:.3f}")

    results_df = pd.DataFrame(results).sort_values("MAE")
    print("\nModel comparison:\n", results_df.to_string(index=False))
    results_df.to_csv(BASE_DIR / "student_regression_comparison.csv", index=False)

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, predictions["Linear Regression"], alpha=0.7,
                label="Linear Regression")
    plt.scatter(y_test, predictions["Random Forest"], alpha=0.7,
                label="Random Forest")
    low = min(y_test.min(), min(pred.min() for pred in predictions.values()))
    high = max(y_test.max(), max(pred.max() for pred in predictions.values()))
    plt.plot([low, high], [low, high], "k--", label="Perfect prediction")
    plt.xlabel("Actual Final Score")
    plt.ylabel("Predicted Final Score")
    plt.title("Actual vs Predicted Student Scores")
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "student_actual_vs_predicted.png", dpi=150)
    plt.close()

    print("\nBest model by lowest MAE:", results_df.iloc[0]["Model"])


if __name__ == "__main__":
    main()
