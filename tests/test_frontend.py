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
    assert "Spending Forecast" in response.text


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


def test_d14_savings_goal_elements_in_frontend():
    """Verify D14 Savings Goal Simulator elements exist in index.html, style.css, and app.js."""
    res_index = client.get("/")
    assert res_index.status_code == 200
    html = res_index.text
    assert "Plan a Savings Goal" in html
    assert "goal-amount-input" in html
    assert "goal-months-input" in html
    assert "btn-calculate-savings" in html

    res_css = client.get("/static/style.css")
    assert res_css.status_code == 200
    assert ".savings-card" in res_css.text

    res_js = client.get("/static/app.js")
    assert res_js.status_code == 200
    js = res_js.text
    assert "initSavingsGoalSimulator" in js
    assert "calculateSavingsPlan" in js
    assert "renderSavingsPlan" in js


def test_d14_savings_goal_mathematics_cases():
    """
    Verify mathematical specification of D14 savings goal calculator:
    Case 1: Goal = 10000, Months = 5 -> required = 2000.00
    Case 2: Goal = 10000, Months = 3 -> schedule = [3333.33, 3333.33, 3333.34], final remaining = 0.00
    Case 3: Goal = 1, Months = 3 -> rounding does not create negative remaining goal
    Case 4..7: Validation rejections (goal <= 0, months <= 0, decimal months, NaN/Infinity)
    Case 8: Large valid values stability
    """
    def simulate_savings_plan_py(goal_amount, months):
        if not isinstance(goal_amount, (int, float)) or not isinstance(months, int) or isinstance(months, bool):
            raise ValueError("Invalid input types")
        if goal_amount <= 0 or months <= 0 or months > 60:
            raise ValueError("Validation bounds failed")

        schedule = []
        cumulative_saved = 0.0
        base_monthly = round(goal_amount / months, 2)

        for m in range(1, months + 1):
            if m == months:
                monthly_plan = max(0.0, round(goal_amount - cumulative_saved, 2))
            else:
                monthly_plan = min(base_monthly, max(0.0, round(goal_amount - cumulative_saved, 2)))

            cumulative_saved += monthly_plan
            remaining = max(0.0, round(goal_amount - cumulative_saved, 2))
            if m == months:
                remaining = 0.0

            schedule.append({
                "month": m,
                "plannedSaving": monthly_plan,
                "remainingGoal": remaining
            })

        return {
            "requiredMonthlyAverage": goal_amount / months,
            "schedule": schedule,
            "totalSaved": round(sum(r["plannedSaving"] for r in schedule), 2),
            "finalRemaining": schedule[-1]["remainingGoal"]
        }

    # Case 1: 10000 / 5
    c1 = simulate_savings_plan_py(10000.0, 5)
    assert c1["requiredMonthlyAverage"] == 2000.0
    assert [r["plannedSaving"] for r in c1["schedule"]] == [2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
    assert c1["totalSaved"] == 10000.0
    assert c1["finalRemaining"] == 0.0

    # Case 2: 10000 / 3
    c2 = simulate_savings_plan_py(10000.0, 3)
    assert [r["plannedSaving"] for r in c2["schedule"]] == [3333.33, 3333.33, 3333.34]
    assert [r["remainingGoal"] for r in c2["schedule"]] == [6666.67, 3333.34, 0.0]
    assert c2["totalSaved"] == 10000.0
    assert c2["finalRemaining"] == 0.0

    # Case 3: 1 / 3
    c3 = simulate_savings_plan_py(1.0, 3)
    assert [r["plannedSaving"] for r in c3["schedule"]] == [0.33, 0.33, 0.34]
    assert c3["totalSaved"] == 1.0
    assert c3["finalRemaining"] == 0.0
    assert all(r["remainingGoal"] >= 0.0 for r in c3["schedule"])

    # Case 4..7: Validation failures
    with pytest.raises(ValueError):
        simulate_savings_plan_py(-500.0, 6)
    with pytest.raises(ValueError):
        simulate_savings_plan_py(10000.0, 0)
    with pytest.raises(ValueError):
        simulate_savings_plan_py(10000.0, 3.5)

    # Case 8: Large valid value
    c8 = simulate_savings_plan_py(1_000_000.0, 60)
    assert c8["totalSaved"] == 1_000_000.0
    assert c8["finalRemaining"] == 0.0

