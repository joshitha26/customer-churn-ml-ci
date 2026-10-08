import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = "customer_churn.csv"


def load_dataset():
    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def main():

    X, y = load_dataset()

    categorical_columns = [
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]

    numerical_columns = [
        "CustomerID",
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns
            )
        ]
    )

    model = LogisticRegression(
        max_iter=1000
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("Customer Churn Model Accuracy:", accuracy)

    joblib.dump(
        pipeline,
        "customer_churn_model.pkl"
    )

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)


if __name__ == "__main__":
    main()
