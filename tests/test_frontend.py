"""
Personal Financial Digital Twin — Web Frontend Test Suite (Dimension 11)

Tests:
1. GET / serves static index.html with HTTP 200.
2. GET /static/style.css serves dark glassmorphism CSS stylesheet with HTTP 200.
3. GET /static/app.js serves frontend client JavaScript application with HTTP 200.
"""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.api.server import app

client = TestClient(app)


def test_frontend_root_index_endpoint():
    """Verify GET / returns HTTP 200 and HTML index document."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "Personal Financial Digital Twin" in response.text
    assert "Explainable Next-Month Spending Forecasting" in response.text


def test_frontend_css_asset():
    """Verify GET /static/style.css serves stylesheet asset with HTTP 200."""
    response = client.get("/static/style.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "") or "text/plain" in response.headers.get("content-type", "")
    assert "--bg-base" in response.text


def test_frontend_js_asset():
    """Verify GET /static/app.js serves JavaScript application script with HTTP 200."""
    response = client.get("/static/app.js")
    assert response.status_code == 200
    assert "javascript" in response.headers.get("content-type", "").lower()
    assert "runPrediction" in response.text
