import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):

        self.assertTrue(
            os.path.exists("customer_churn.csv")
        )

    def test_model_created(self):

        self.assertTrue(
            os.path.exists("customer_churn_model.pkl")
        )

    def test_metrics_created(self):

        self.assertTrue(
            os.path.exists("metrics.json")
        )

    def test_accuracy_is_valid(self):

        with open("metrics.json", "r") as file:

            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(
            accuracy,
            0.0
        )

        self.assertLessEqual(
            accuracy,
            1.0
        )

    def test_model_prediction(self):

        model = joblib.load(
            "customer_churn_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 35,
            "Gender": "Female",
            "Tenure": 20,
            "Usage Frequency": 15,
            "Support Calls": 3,
            "Payment Delay": 5,
            "Subscription Type": "Standard",
            "Contract Length": "Annual",
            "Total Spend": 600,
            "Last Interaction": 10
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )

    def test_second_customer_prediction(self):

        model = joblib.load(
            "customer_churn_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 45,
            "Gender": "Male",
            "Tenure": 30,
            "Usage Frequency": 25,
            "Support Calls": 1,
            "Payment Delay": 2,
            "Subscription Type": "Premium",
            "Contract Length": "Annual",
            "Total Spend": 900,
            "Last Interaction": 5
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )

    def test_low_engagement_customer_prediction(self):

        model = joblib.load(
            "customer_churn_model.pkl"
        )

        sample = pd.DataFrame([{
            "Age": 50,
            "Gender": "Female",
            "Tenure": 5,
            "Usage Frequency": 5,
            "Support Calls": 10,
            "Payment Delay": 25,
            "Subscription Type": "Basic",
            "Contract Length": "Monthly",
            "Total Spend": 200,
            "Last Interaction": 25
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(
            int(prediction),
            [0, 1]
        )


if __name__ == "__main__":
    unittest.main()
