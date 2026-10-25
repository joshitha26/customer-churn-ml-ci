from pathlib import Path
import logging
import os
import sys
import time
import uuid

import joblib
import pandas as pd

from flask import Flask, jsonify, request, g
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


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

MODEL_PATH = Path(__file__).resolve().parent / "customer_churn_model.pkl"

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
# 4. Prometheus Monitoring Metrics
# --------------------------------------------------

# Count HTTP requests by method, endpoint, and status.
HTTP_REQUESTS = Counter(
    "customer_churn_http_requests_total",
    "Total HTTP requests received by the application",
    ["method", "endpoint", "status"],
)

# Measure HTTP request processing duration.
HTTP_REQUEST_DURATION = Histogram(
    "customer_churn_http_request_duration_seconds",
    "HTTP request processing duration in seconds",
    ["method", "endpoint"],
)

# Count prediction requests by prediction outcome.
PREDICTIONS_TOTAL = Counter(
    "customer_churn_predictions_total",
    "Total successful customer churn predictions",
    ["prediction"],
)

# Track the number of currently active requests.
ACTIVE_REQUESTS = Gauge(
    "customer_churn_active_requests",
    "Number of HTTP requests currently being processed",
)


# --------------------------------------------------
# 5. Load Trained Model
# --------------------------------------------------

def load_model():
    """Load the trained customer churn model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH.name} was not found. "
            "Ensure the trained model is included in the deployment."
        )

    return joblib.load(MODEL_PATH)


# --------------------------------------------------
# 6. Log Incoming Requests
# --------------------------------------------------

@app.before_request
def log_request_start():
    """Record the request ID, start time, and active request count."""

    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4()),
    )

    g.start_time = time.perf_counter()
    ACTIVE_REQUESTS.inc()

    logger.info(
        "event=request_received request_id=%s method=%s path=%s",
        g.request_id,
        request.method,
        request.path,
    )


# --------------------------------------------------
# 7. Log Responses and Update HTTP Metrics
# --------------------------------------------------

@app.after_request
def log_request_end(response):
    """Record response status, duration, and request metrics."""

    start_time = getattr(g, "start_time", time.perf_counter())

    duration_seconds = time.perf_counter() - start_time
    duration_ms = duration_seconds * 1000

    request_id = getattr(g, "request_id", "unknown")

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

    # Do not count the metrics endpoint itself.
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint=request.path,
            status=str(response.status_code),
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            endpoint=request.path,
        ).observe(duration_seconds)

    response.headers["X-Request-ID"] = request_id

    ACTIVE_REQUESTS.dec()

    return response


# --------------------------------------------------
# 8. Global Error Tracking
# --------------------------------------------------

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Log unexpected errors and return a safe response."""

    logger.exception(
        "event=unexpected_error request_id=%s error_type=%s",
        getattr(g, "request_id", "unknown"),
        type(error).__name__,
    )

    return jsonify({
        "error": "Internal server error",
        "request_id": getattr(g, "request_id", "unknown"),
    }), 500


# --------------------------------------------------
# 9. Health Check Endpoint
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
# 10. Prometheus Metrics Endpoint
# --------------------------------------------------

@app.get("/metrics")
def metrics():
    """Expose application metrics in Prometheus format."""

    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST
    }


# --------------------------------------------------
# 11. Customer Churn Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict():
    """Predict whether a customer is likely to churn."""

    request_id = getattr(g, "request_id", "unknown")

    logger.info(
        "event=prediction_request request_id=%s",
        request_id,
    )

    # Validate the JSON request body.
    data = request.get_json(silent=True)

    if not isinstance(data, dict) or not data:
        logger.warning(
            "event=invalid_request request_id=%s",
            request_id,
        )

        return jsonify({
            "error": "A valid JSON request body is required",
            "request_id": request_id,
        }), 400

    # Check required customer features.
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=missing_fields request_id=%s fields=%s",
            request_id,
            ",".join(missing_fields),
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields,
            "request_id": request_id,
        }), 400

    try:
        # Prepare the input in the model's expected feature order.
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        # Load the trained model.
        model = load_model()

        # Generate prediction.
        prediction_code = int(model.predict(sample)[0])

        prediction = (
            "CHURN"
            if prediction_code == 1
            else "NO CHURN"
        )

        # Update prediction metrics.
        PREDICTIONS_TOTAL.labels(
            prediction=prediction
        ).inc()

        # Log the result without logging customer input data.
        logger.info(
            (
                "event=prediction_success "
                "request_id=%s prediction=%s prediction_code=%s"
            ),
            request_id,
            prediction,
            prediction_code,
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code,
            "request_id": request_id,
        }), 200

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            request_id,
        )

        return jsonify({
            "error": "Prediction failed",
            "request_id": request_id,
        }), 500


# --------------------------------------------------
# 12. Start Flask Application
# --------------------------------------------------

if __name__ == "__main__":
    logger.info("event=application_starting")

    # Render provides PORT; local execution defaults to 5000.
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
