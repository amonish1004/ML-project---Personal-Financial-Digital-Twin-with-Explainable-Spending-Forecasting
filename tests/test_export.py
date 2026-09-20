"""
Personal Financial Digital Twin — Export Service Unit Tests

Tests:
1. PDF report generation with complete export state payload.
2. PDF report generation with minimal baseline payload (missing optional sections).
3. Excel workbook generation with complete export state payload.
4. Excel workbook generation with minimal baseline payload.
5. Content validity assertions (%PDF magic header, openpyxl workbook structure).
6. Currency-agnostic assertions verifying CZK wording is completely absent.
"""

import sys
import io
from pathlib import Path
import pytest
import openpyxl

# Resolve project root portably
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.export import generate_pdf_report, generate_excel_report

SAMPLE_BASELINE_STATE = {
    "ending_balance_t": 15000.0,
    "income_credit_t": 25000.0,
    "debit_count_t": 12,
    "spending_hh_t": 3000.0,
    "spending_st_t": 500.0,
    "spending_in_t": 1200.0,
    "spending_lo_t": 2000.0,
    "spending_io_t": 300.0,
    "spending_other_t": 1500.0,
    "spending_t_minus_1": 8000.0,
    "spending_t_minus_2": 7500.0,
}

SAMPLE_COMPLETE_EXPORT_PAYLOAD = {
    "generated_at": "2026-09-20 19:30:00",
    "baseline_state": SAMPLE_BASELINE_STATE,
    "prediction": 8250.50,
    "scenario_data": {
        "basePred": 8250.50,
        "scenPred": 9100.00,
        "absDiff": 849.50,
        "pctDiff": 10.30,
        "fieldName": "Household Spending",
        "newVal": 5000.0,
    },
    "savings_plan": {
        "rawName": "Emergency Fund",
        "amount": 20000.0,
        "months": 6,
        "requiredMonthly": 3333.33,
        "schedule": [
            {"month": 1, "plannedSaving": 3333.33, "remainingGoal": 16666.67},
            {"month": 2, "plannedSaving": 3333.33, "remainingGoal": 13333.34},
        ],
    },
    "shap_data": {
        "base_value": 7800.0,
        "reconstructed_prediction": 8250.50,
        "additivity_delta": 0.000001,
        "features": [
            {"feature": "spending_hh_t", "value": 3000.0, "shap_value": 250.0},
            {"feature": "income_credit_t", "value": 25000.0, "shap_value": -100.0},
        ],
    },
}


def test_generate_pdf_report_complete():
    """Verify PDF generation returns valid binary starting with PDF header and omitting currency labels."""
    pdf_bytes = generate_pdf_report(SAMPLE_COMPLETE_EXPORT_PAYLOAD)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 1000
    assert pdf_bytes.startswith(b"%PDF")
    assert b"CZK" not in pdf_bytes


def test_generate_pdf_report_minimal():
    """Verify PDF generation handles minimal payload without optional sections gracefully and without currency labels."""
    minimal_payload = {
        "baseline_state": SAMPLE_BASELINE_STATE,
        "prediction": 8250.50,
    }
    pdf_bytes = generate_pdf_report(minimal_payload)
    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 500
    assert pdf_bytes.startswith(b"%PDF")
    assert b"CZK" not in pdf_bytes


def test_generate_excel_report_complete():
    """Verify Excel generation produces readable openpyxl workbook with all worksheets and no CZK wording."""
    excel_bytes = generate_excel_report(SAMPLE_COMPLETE_EXPORT_PAYLOAD)
    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 1000
    assert b"CZK" not in excel_bytes

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Executive Summary" in sheet_names
    assert "Financial Inputs" in sheet_names
    assert "Forecast & Scenario" in sheet_names
    assert "Savings Goal Schedule" in sheet_names
    assert "SHAP Attributions" in sheet_names

    ws_sum = wb["Executive Summary"]
    assert ws_sum["A1"].value == "Personal Financial Digital Twin — Executive Summary"
    assert "CZK" not in str(ws_sum["A2"].value)
    assert ws_sum["B5"].value == 8250.50
    assert ws_sum["B6"].value == 15000.00

    ws_sav = wb["Savings Goal Schedule"]
    assert "Emergency Fund" in str(ws_sav["A1"].value)
    assert ws_sav["B4"].value == 20000.00
    assert ws_sav["B5"].value == 6
    assert ws_sav["B6"].value == 3333.33


def test_generate_excel_report_minimal():
    """Verify Excel generation handles minimal payload omitting optional worksheets and without CZK wording."""
    minimal_payload = {
        "baseline_state": SAMPLE_BASELINE_STATE,
        "prediction": 8250.50,
    }
    excel_bytes = generate_excel_report(minimal_payload)
    assert isinstance(excel_bytes, bytes)
    assert b"CZK" not in excel_bytes

    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    sheet_names = wb.sheetnames
    assert "Executive Summary" in sheet_names
    assert "Financial Inputs" in sheet_names
    assert "Forecast & Scenario" not in sheet_names
    assert "Savings Goal Schedule" not in sheet_names
    assert "SHAP Attributions" not in sheet_names


def test_scenario_zero_value_handling():
    """Verify numeric scenario values with 0.0 are preserved instead of triggering fallback."""
    zero_payload = {
        "baseline_state": SAMPLE_BASELINE_STATE,
        "prediction": 8250.50,
        "scenario_data": {
            "basePred": 8250.50,
            "scenPred": 0.0,
            "absDiff": 0.0,
            "pctDiff": 0.0,
            "baseline_prediction": 9999.99,
            "scenario_prediction": 9999.99,
            "absolute_difference": 9999.99,
            "percentage_difference": 9999.99,
        },
    }
    excel_bytes = generate_excel_report(zero_payload)
    wb = openpyxl.load_workbook(io.BytesIO(excel_bytes))
    ws_scen = wb["Forecast & Scenario"]
    assert ws_scen["C4"].value == 0.0
    assert ws_scen["D4"].value == 0.0

