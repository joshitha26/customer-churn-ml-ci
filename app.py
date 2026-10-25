from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request


# --------------------------------------------------
# 1. Initialize Flask Application
# --------------------------------------------------

app = Flask(__name__)


# --------------------------------------------------
# 2. Configure Application Logging
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("customer-churn-api")


# --------------------------------------------------
# 3. Model Configuration
# --------------------------------------------------

MODEL_PATH = Path("customer_churn_model.pkl")

FEATURES = [
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
]


# --------------------------------------------------
# 4. Load Trained Model
# --------------------------------------------------

def load_model():
    """Load the trained customer churn model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "customer_churn_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 5. Log Incoming Requests
# --------------------------------------------------

@app.before_request
def log_request_start():
    """Record request ID and start time."""

    request.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    request.start_time = time.perf_counter()

    logger.info(
        "event=request_received request_id=%s method=%s path=%s",
        request.request_id,
        request.method,
        request.path,
    )


# --------------------------------------------------
# 6. Log Responses and Response Time
# --------------------------------------------------

@app.after_request
def log_request_end(response):
    """Record HTTP status and request duration."""

    start_time = getattr(
        request,
        "start_time",
        time.perf_counter(),
    )

    duration_ms = (
        time.perf_counter() - start_time
    ) * 1000

    request_id = getattr(
        request,
        "request_id",
        "unknown",
    )

    if response.status_code >= 500:
        log_level = logging.ERROR
    elif response.status_code >= 400:
        log_level = logging.WARNING
    else:
        log_level = logging.INFO

    logger.log(
        log_level,
        (
            "event=http_request_completed "
            "request_id=%s method=%s path=%s "
            "status=%s duration_ms=%.2f"
        ),
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id

    return response


# --------------------------------------------------
# 7. Global Error Tracking
# --------------------------------------------------

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Log unexpected errors and return a safe response."""

    logger.exception(
        "event=unexpected_error request_id=%s error_type=%s",
        getattr(request, "request_id", "unknown"),
        type(error).__name__,
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(
            request,
            "request_id",
            "unknown",
        ),
    }), 500


# --------------------------------------------------
# 8. Health Check Endpoint
# --------------------------------------------------

@app.get("/")
def health_check():
    """Check whether the application is running."""

    logger.info("event=health_check")

    return jsonify({
        "status": "ok",
        "service": "customer-churn-prediction",
    })


# --------------------------------------------------
# 9. Customer Churn Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    """Predict whether a customer is likely to churn."""

    logger.info(
        "event=prediction_request request_id=%s",
        request.request_id,
    )

    # Read JSON input
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        logger.warning(
            "event=invalid_request request_id=%s reason=invalid_or_missing_json",
            request.request_id,
        )

        return jsonify({
            "error": "A valid JSON request body is required",
            "request_id": request.request_id,
        }), 400

    # Check required customer features
    missing_fields = [
        feature
        for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=missing_fields request_id=%s fields=%s",
            request.request_id,
            ",".join(missing_fields),
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
            "request_id": request.request_id,
        }), 400

    try:
        # Prepare input using the existing feature order
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        # Load the trained model
        model = load_model()

        # Generate prediction
        prediction_code = int(
            model.predict(sample)[0]
        )

        prediction = (
            "CHURN"
            if prediction_code == 1
            else "NO CHURN"
        )

        # Log successful prediction without logging customer data
        logger.info(
            (
                "event=prediction_success "
                "request_id=%s prediction=%s prediction_code=%s"
            ),
            request.request_id,
            prediction,
            prediction_code,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
            "request_id": request.request_id,
        })

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request.request_id,
        )

        return jsonify({
            "error": "Prediction failed",
            "request_id": request.request_id,
        }), 500


# --------------------------------------------------
# 10. Start Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )

