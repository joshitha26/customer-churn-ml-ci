import json
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = "customer_churn.csv"


def load_dataset():
    print("Loading customer churn dataset...")

    data = pd.read_csv(DATA_FILE)

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))
    print("Number of columns:", len(data.columns))

    return data


def train_model():

    data = load_dataset()

    # Remove CustomerID because it is only an identifier
    data = data.drop(columns=["CustomerID"])

    features = [
        "Age",
        "Gender",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Subscription Type",
        "Contract Length",
        "Total Spend",
        "Last Interaction"
    ]

    target = "Churn"

    X = data[features]
    y = data[target]

    print("Target distribution:")
    print(y.value_counts())

    # Numerical and categorical columns
    numerical_features = [
        "Age",
        "Tenure",
        "Usage Frequency",
        "Support Calls",
        "Payment Delay",
        "Total Spend",
        "Last Interaction"
    ]

    categorical_features = [
        "Gender",
        "Subscription Type",
        "Contract Length"
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    # ML pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    print("Training customer churn model...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    # Save model
    joblib.dump(
        model,
        "customer_churn_model.pkl"
    )

    print(
        "\nModel saved as customer_churn_model.pkl"
    )

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    print("Metrics saved as metrics.json")


if __name__ == "__main__":
    train_model()
