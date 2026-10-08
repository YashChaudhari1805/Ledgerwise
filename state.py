"""Pydantic schemas and the LangGraph shared state."""
import operator
from datetime import date
from typing import Annotated, Any, Literal, Optional, TypedDict

from pydantic import BaseModel, ConfigDict, Field, field_validator


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: str
    quantity: float = Field(ge=0, description="Billed hours")
    unit_rate: float = Field(ge=0, description="USD per hour")
    line_total_claimed: float


class InvoiceDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor_name: str
    vendor_id: str
    invoice_number: str
    invoice_date: date
    line_items: list[LineItem] = Field(min_length=1)
    subtotal_claimed: float
    discount_claimed: float = 0.0
    tax_rate: float = Field(description="Fraction, e.g. 0.08 (8 is accepted and normalised)")
    tax_claimed: float
    total_claimed: float

    @field_validator("tax_rate")
    @classmethod
    def _normalise_rate(cls, v: float) -> float:
        return v / 100 if v > 1 else v


Category = Literal["MATH_ERROR", "RATE_CREEP", "HOURS_CAP", "SUBTOTAL_MISMATCH",
                   "DISCOUNT_ERROR", "TAX_ERROR", "TOTAL_MISMATCH", "UNKNOWN_ITEM"]
Classification = Literal["Contractual Violation", "Informational: Rounding Variance",
                         "Needs Manual Review"]


class Discrepancy(BaseModel):
    id: str
    category: Category
    line_ref: str
    description: str
    billed: float
    expected: float
    delta: float = Field(description="Positive = vendor overcharge (billed - calculated)")
    clause: Optional[str] = None
    clause_text: Optional[str] = None
    classification: Optional[Classification] = None
    rationale: Optional[str] = None


class ToolTrace(BaseModel):
    agent: str
    tool: str
    args: dict[str, Any] = {}
    result: Any = None
    ms: float = 0.0


class AuditState(TypedDict, total=False):
    # inputs
    vendor_id: str
    raw_invoice: str
    # ingestion
    invoice: InvoiceDocument
    ingestion_method: str
    # auditor
    findings: list[Discrepancy]
    expected: dict[str, float]
    # critic
    actionable: list[Discrepancy]
    suppressed: list[Discrepancy]
    needs_reaudit: bool
    reaudit_count: int
    verdict: Literal["APPROVED_FOR_PAYMENT", "VIOLATIONS_FOUND"]
    # resolution
    report_md: str
    dispute_email: str
    ledger_record: dict[str, Any]
    # shared, append-only trace log (reducer concatenates across nodes)
    traces: Annotated[list[ToolTrace], operator.add]
