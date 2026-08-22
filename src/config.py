import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "berka"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Ensure output directories exist
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Explicit schemas for headerless TSV raw files
SCHEMAS = {
    "fin_trans.tsv": [
        "trans_id", "account_id", "date", "amount", "balance",
        "type", "operation", "k_symbol", "bank", "account"
    ],
    "fin_account.tsv": [
        "account_id", "district_id", "date", "frequency"
    ],
    "fin_client.tsv": [
        "client_id", "birth_date", "sex", "district_id"
    ],
    "fin_disp.tsv": [
        "disp_id", "client_id", "account_id", "type"
    ],
    "fin_order.tsv": [
        "order_id", "account_id", "bank_to", "account_to", "amount", "k_symbol"
    ],
    "fin_loan.tsv": [
        "loan_id", "account_id", "date", "amount", "duration", "payments", "status"
    ],
    "fin_card.tsv": [
        "card_id", "disp_id", "type", "issued"
    ],
    "fin_district.tsv": [
        "district_id", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9",
        "A10", "A11", "A12", "A13", "A14", "A15", "A16"
    ]
}

# Transaction type codes
TYPE_DEBIT = "D"
TYPE_CREDIT = "C"
TYPE_INTEREST = "P"

# Category symbols for debits
K_SYMBOL_HOUSEHOLD = "HH"
K_SYMBOL_STATEMENT = "ST"
K_SYMBOL_INSURANCE = "IN"
K_SYMBOL_LOAN = "LO"
K_SYMBOL_INTEREST_OUT = "IO"
