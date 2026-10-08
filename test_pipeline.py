"""Agentic acceptance tests (run offline: no API key needed).  `pytest -q`  or  `python test_pipeline.py`"""
import os, tempfile
os.environ["AUDIT_LEDGER_PATH"] = os.path.join(tempfile.mkdtemp(), "ledger.jsonl")
from pathlib import Path
from graph import LLMClient, run_audit

DATA = Path(__file__).parent / "test_data"
OFFLINE = LLMClient("offline")


def audit(name):
    return run_audit((DATA / name).read_text(), "VENDOR-A", OFFLINE)


def by_cat(rows, cat):
    return [f for f in rows if f.category == cat]


def test_clean_invoice_is_approved():
    s = audit("invoice_clean.txt")
    assert s["verdict"] == "APPROVED_FOR_PAYMENT" and not s["actionable"]
    assert s["ledger_record"]["amount_approved"] == 9936.00 and s["ledger_record"]["due_date"] == "2026-10-30"


def test_math_trap_and_penny_drift():
    s = audit("invoice_math_err.txt")
    math = by_cat(s["actionable"], "MATH_ERROR")
    assert len(math) == 1 and math[0].expected == 1200.00 and math[0].delta == 200.00
    assert [f.delta for f in s["suppressed"]] == [0.02]
    assert s["suppressed"][0].classification == "Informational: Rounding Variance"
    assert s["expected"]["total"] == 6045.30


def test_rate_creep_cap_and_pre_discount_tax():
    s = audit("invoice_creep.txt")
    a = s["actionable"]
    assert by_cat(a, "RATE_CREEP")[0].delta == 800.00
    cap = by_cat(a, "HOURS_CAP")[0]
    assert cap.delta == 2125.00 and "25 hrs" in cap.description
    tax = by_cat(a, "TAX_ERROR")[0]
    assert tax.delta == 66.10 and "PRE-discount" in tax.description
    assert s["expected"]["total"] == 13953.60
    assert "§3.1" in s["dispute_email"] and "§5.1" in s["dispute_email"]


def test_reflection_loop_for_unmapped_item():
    s = audit("invoice_unmapped.txt")
    assert s["reaudit_count"] == 1
    unk = by_cat(s["actionable"], "UNKNOWN_ITEM")[0]
    assert unk.classification == "Needs Manual Review" and s["verdict"] == "VIOLATIONS_FOUND"


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn(); print("PASS", n)
