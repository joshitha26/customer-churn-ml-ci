import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):

        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            response.get_json()["status"],
            "ok"
        )

    def test_prediction_endpoint(self):

        response = self.client.post(
            "/predict",
            json={
                "Age": 35,
                "Gender": "Male",
                "Tenure": 24,
                "Usage Frequency": 20,
                "Support Calls": 2,
                "Payment Delay": 5,
                "Subscription Type": "Premium",
                "Contract Length": "Annual",
                "Total Spend": 1200,
                "Last Interaction": 10
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )

        result = response.get_json()

        self.assertIn(
            result["prediction_code"],
            [0, 1]
        )

        self.assertIn(
            result["prediction"],
            ["CHURN", "NO CHURN"]
        )

    def test_missing_field_validation(self):

        response = self.client.post(
            "/predict",
            json={
                "Age": 35,
                "Gender": "Male"
            }
        )

        self.assertEqual(
            response.status_code,
            400
        )

        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
