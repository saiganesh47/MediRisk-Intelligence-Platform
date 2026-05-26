"""
MediRisk Intelligence Platform
Synthetic Data Generator
Generates realistic Medicare claims, patient records, and financial transactions
with injected fraud patterns for model training.
"""

import json
import random
import uuid
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

fake = Faker()
random.seed(42)

# ─── Constants ────────────────────────────────────────────────────────────────

DIAGNOSIS_CODES = [
    "I10", "E11.9", "J18.9", "N18.3", "I25.10", "F32.9", "M54.5",
    "Z87.891", "I48.91", "E78.5", "K21.0", "J44.1", "G47.33", "E03.9",
]
PROCEDURE_CODES = [
    "99213", "99214", "99215", "93000", "85025", "80053", "71046",
    "93306", "43239", "45378", "70553", "27447", "33533", "36415",
]
PROVIDER_SPECIALTIES = [
    "General Practice", "Cardiology", "Oncology", "Neurology",
    "Orthopedics", "Psychiatry", "Internal Medicine", "Emergency Medicine",
]
INSURANCE_TYPES = ["Medicare", "Medicaid", "BlueCross", "Aetna", "UnitedHealth", "Cigna"]
CLAIM_STATUSES   = ["SUBMITTED", "APPROVED", "DENIED", "PENDING", "ADJUSTED"]

FRAUD_PATTERNS = [
    "upcoding",           # billing higher-complexity procedures
    "phantom_billing",    # billing for services not rendered
    "duplicate_claim",    # same claim submitted multiple times
    "unbundling",         # splitting bundled services to inflate payment
    "kickback",           # referral fraud
]


# ─── Generators ───────────────────────────────────────────────────────────────

def generate_patients(n: int = 5_000) -> pd.DataFrame:
    records = []
    for _ in range(n):
        dob = fake.date_of_birth(minimum_age=18, maximum_age=90)
        records.append({
            "patient_id":        str(uuid.uuid4()),
            "first_name":        fake.first_name(),
            "last_name":         fake.last_name(),
            "dob":               dob.isoformat(),
            "gender":            random.choice(["M", "F", "O"]),
            "zip_code":          fake.zipcode(),
            "state":             fake.state_abbr(),
            "insurance_type":    random.choice(INSURANCE_TYPES),
            "insurance_id":      fake.bothify(text="???-########"),
            "chronic_conditions": random.randint(0, 6),
            "created_at":        fake.date_time_between(
                                     start_date="-5y", end_date="now"
                                 ).isoformat(),
        })
    return pd.DataFrame(records)


def generate_providers(n: int = 500) -> pd.DataFrame:
    records = []
    for _ in range(n):
        is_flagged = random.random() < 0.05   # 5% are known bad actors
        records.append({
            "provider_id":   str(uuid.uuid4()),
            "npi":           fake.numerify(text="##########"),
            "name":          fake.company() + " Medical Group",
            "specialty":     random.choice(PROVIDER_SPECIALTIES),
            "state":         fake.state_abbr(),
            "zip_code":      fake.zipcode(),
            "is_flagged":    is_flagged,
            "license_valid": not is_flagged or random.random() < 0.5,
            "joined_date":   fake.date_between(start_date="-10y", end_date="-1y").isoformat(),
        })
    return pd.DataFrame(records)


def generate_claims(
    patients: pd.DataFrame,
    providers: pd.DataFrame,
    n: int = 100_000,
) -> pd.DataFrame:
    patient_ids  = patients["patient_id"].tolist()
    provider_ids = providers["provider_id"].tolist()
    flagged_providers = set(
        providers.loc[providers["is_flagged"], "provider_id"].tolist()
    )

    records = []
    seen_claim_keys: set = set()

    for _ in range(n):
        provider_id  = random.choice(provider_ids)
        patient_id   = random.choice(patient_ids)
        service_date = fake.date_between(start_date="-2y", end_date="today")
        billed_amt   = round(random.uniform(50, 15_000), 2)
        allowed_amt  = round(billed_amt * random.uniform(0.4, 0.95), 2)
        paid_amt     = round(allowed_amt * random.uniform(0.7, 1.0), 2)

        # Inject fraud signals
        is_fraud      = False
        fraud_type    = None
        fraud_signals = []

        if provider_id in flagged_providers:
            fraud_roll = random.random()

            if fraud_roll < 0.30:
                fraud_type = "upcoding"
                billed_amt = round(billed_amt * random.uniform(2.5, 5.0), 2)
                fraud_signals.append("HIGH_BILLED_RATIO")
                is_fraud = True

            elif fraud_roll < 0.50:
                fraud_type = "duplicate_claim"
                fraud_signals.append("DUPLICATE_SUBMISSION")
                is_fraud = True

            elif fraud_roll < 0.65:
                fraud_type = "phantom_billing"
                fraud_signals.append("NO_MATCHING_ENCOUNTER")
                is_fraud = True

            elif fraud_roll < 0.75:
                fraud_type = "unbundling"
                fraud_signals.append("UNBUNDLED_CODES")
                is_fraud = True

        # Generic anomaly signals
        if billed_amt > 10_000:
            fraud_signals.append("HIGH_VALUE_CLAIM")
        if random.random() < 0.02:
            fraud_signals.append("WEEKEND_SUBMISSION")

        claim_key = f"{patient_id}:{provider_id}:{service_date}"
        is_duplicate = claim_key in seen_claim_keys
        if not is_duplicate:
            seen_claim_keys.add(claim_key)

        records.append({
            "claim_id":        str(uuid.uuid4()),
            "patient_id":      patient_id,
            "provider_id":     provider_id,
            "service_date":    service_date.isoformat(),
            "submitted_date":  (service_date + timedelta(days=random.randint(1, 30))).isoformat(),
            "diagnosis_code":  random.choice(DIAGNOSIS_CODES),
            "procedure_code":  random.choice(PROCEDURE_CODES),
            "billed_amount":   billed_amt,
            "allowed_amount":  allowed_amt,
            "paid_amount":     paid_amt,
            "claim_status":    random.choice(CLAIM_STATUSES),
            "is_duplicate":    is_duplicate,
            "is_fraud":        is_fraud,
            "fraud_type":      fraud_type,
            "fraud_signals":   json.dumps(fraud_signals),
            "ingestion_ts":    datetime.utcnow().isoformat(),
        })

    return pd.DataFrame(records)


def generate_financial_transactions(
    patients: pd.DataFrame,
    n: int = 50_000,
) -> pd.DataFrame:
    patient_ids = patients["patient_id"].tolist()
    tx_types = ["COPAY", "DEDUCTIBLE", "PREMIUM", "REFUND", "ADJUSTMENT", "OVERPAYMENT"]

    records = []
    for _ in range(n):
        amount = round(random.uniform(5, 5_000), 2)
        is_anomalous = random.random() < 0.03  # 3% anomalous
        if is_anomalous:
            amount = round(random.uniform(10_000, 100_000), 2)

        records.append({
            "transaction_id":   str(uuid.uuid4()),
            "patient_id":       random.choice(patient_ids),
            "transaction_type": random.choice(tx_types),
            "amount":           amount,
            "transaction_date": fake.date_between(start_date="-2y", end_date="today").isoformat(),
            "channel":          random.choice(["ONLINE", "PHONE", "IN_PERSON", "MAIL"]),
            "is_anomalous":     is_anomalous,
            "ingestion_ts":     datetime.utcnow().isoformat(),
        })

    return pd.DataFrame(records)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(output_dir: str = "data/synthetic/output"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("⏳ Generating patients...")
    patients = generate_patients(5_000)
    patients.to_csv(f"{output_dir}/patients.csv", index=False)
    print(f"   ✅ {len(patients):,} patients written")

    print("⏳ Generating providers...")
    providers = generate_providers(500)
    providers.to_csv(f"{output_dir}/providers.csv", index=False)
    print(f"   ✅ {len(providers):,} providers written")

    print("⏳ Generating claims...")
    claims = generate_claims(patients, providers, 100_000)
    claims.to_parquet(f"{output_dir}/claims.parquet", index=False)
    fraud_rate = claims["is_fraud"].mean() * 100
    print(f"   ✅ {len(claims):,} claims written  |  Fraud rate: {fraud_rate:.2f}%")

    print("⏳ Generating financial transactions...")
    transactions = generate_financial_transactions(patients, 50_000)
    transactions.to_parquet(f"{output_dir}/transactions.parquet", index=False)
    print(f"   ✅ {len(transactions):,} transactions written")

    # Schema snapshot
    schema = {
        "patients":     list(patients.columns),
        "providers":    list(providers.columns),
        "claims":       list(claims.columns),
        "transactions": list(transactions.columns),
    }
    with open(f"{output_dir}/schema.json", "w") as f:
        json.dump(schema, f, indent=2)

    print(f"\n🎉 All data written to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediRisk synthetic data generator")
    parser.add_argument("--output-dir", default="data/synthetic/output")
    args = parser.parse_args()
    main(args.output_dir)
