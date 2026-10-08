"""Deterministic tools. All money maths happens here, never in an LLM prompt."""
import re
import time
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
from pathlib import Path
from typing import Any

from state import ToolTrace

DATA_DIR = Path(__file__).parent / "test_data"


def _q(x: Any) -> float:
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def _D(x: Any) -> Decimal:
    return Decimal(str(x))


# ----------------------------------------------------------------- arithmetic
def calc_line_item(qty: float, unit_rate: float) -> float:
    """Exact qty x rate, rounded half-up to cents."""
    return _q(_D(qty) * _D(unit_rate))


def calc_tax(subtotal: float, tax_rate: float) -> float:
    """Expected tax. tax_rate is a fraction (0.08); values > 1 are treated as percent."""
    rate = _D(tax_rate) / 100 if tax_rate > 1 else _D(tax_rate)
    return _q(_D(subtotal) * rate)


def calc_total(subtotal: float, tax: float, discounts: float = 0.0) -> float:
    """subtotal - discounts + tax."""
    return _q(_D(subtotal) - _D(discounts) + _D(tax))


def calc_sum(values: list) -> float:
    return _q(sum((_D(v) for v in values), Decimal(0)))


def calc_discount(subtotal: float, threshold: float, pct: float) -> float:
    """Volume discount: pct% of subtotal when subtotal exceeds threshold, else 0."""
    return _q(_D(subtotal) * _D(pct) / 100) if subtotal > threshold else 0.0


def check_tolerance_threshold(delta: float, threshold: float = 1.00) -> bool:
    """True if |delta| is MATERIAL (exceeds threshold); False if within rounding variance."""
    return _q(abs(_D(delta))) > _D(threshold)


# ------------------------------------------------------------------- contract
@lru_cache(maxsize=8)
def load_contract(vendor_id: str) -> dict:
    slug = re.sub(r"[^a-z0-9]+", "_", vendor_id.strip().lower())
    path = DATA_DIR / f"contract_{slug}.md"
    if not path.exists():
        raise FileNotFoundError(f"No contract on file for vendor '{vendor_id}' ({path.name})")
    text = path.read_text(encoding="utf-8")
    rates, terms, clauses = {}, {}, {}
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("|") and not set(s.replace(" ", "")) <= set("|-:"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if len(cells) == 6 and cells[0] != "item_key":
                key, desc, rate, cap, kws, clause = cells
                rates[key] = {"description": desc, "rate": float(rate), "cap": float(cap),
                              "keywords": [k.strip().lower() for k in kws.split(",")], "clause": clause}
        if m := re.match(r"^- `(\w+)`:\s*(.+?)\s*\[§(\d+\.\d+)\]\s*$", s):
            raw = m.group(2)
            try:
                val: Any = float(raw)
            except ValueError:
                val = raw
            terms[m.group(1)] = (val, m.group(3))
        if m := re.match(r"^\*\*(\d+\.\d+)\s+([^*]+?)\.?\*\*\s*(.+)$", s):
            clauses[m.group(1)] = (m.group(2).strip(), m.group(3).strip())
    return {"rates": rates, "terms": terms, "clauses": clauses}


def _cl(c: dict, num: str) -> dict:
    title, text = c["clauses"].get(num, ("", ""))
    return {"clause": f"§{num}", "clause_title": title, "clause_text": text}


def lookup_contract_clause(vendor_id: str, item_key: str) -> dict:
    """Rates, hour caps, discounts, tax rules, payment terms, tolerance for a vendor."""
    try:
        c = load_contract(vendor_id)
    except FileNotFoundError as e:
        return {"found": False, "error": str(e)}
    key = item_key.strip().lower()
    t = c["terms"]
    if key in c["rates"]:
        r = c["rates"][key]
        return {"found": True, "item_key": key, "description": r["description"],
                "rate_usd_per_hr": r["rate"], "monthly_cap_hours": r["cap"],
                "payment_terms": t.get("payment_terms", ("", ""))[0], **_cl(c, r["clause"])}
    if key == "volume_discount":
        return {"found": True, "item_key": key, "threshold_usd": t["volume_discount_threshold_usd"][0],
                "pct": t["volume_discount_pct"][0], **_cl(c, t["volume_discount_pct"][1])}
    if key == "tax":
        return {"found": True, "item_key": key, "rate": t["tax_rate_pct"][0] / 100,
                "basis": t["tax_basis"][0], **_cl(c, t["tax_rate_pct"][1])}
    if key in t:
        return {"found": True, "item_key": key, "value": t[key][0], **_cl(c, t[key][1])}
    return {"found": False, "item_key": key,
            "available_keys": sorted(c["rates"]) + ["volume_discount", "tax"] + sorted(t)}


def resolve_item_key(vendor_id: str, description: str) -> dict:
    """Deterministically map an invoice line description to a contract item_key."""
    try:
        c = load_contract(vendor_id)
    except FileNotFoundError:
        return {"item_key": None, "matched_keyword": None}
    best = (None, None)
    for key, r in c["rates"].items():
        for kw in r["keywords"]:
            if re.search(rf"\b{re.escape(kw)}\b", description.lower()):
                if best[1] is None or len(kw) > len(best[1]):
                    best = (key, kw)
    return {"item_key": best[0], "matched_keyword": best[1]}


TOOL_REGISTRY = {f.__name__: f for f in (
    calc_line_item, calc_tax, calc_total, calc_sum, calc_discount,
    check_tolerance_threshold, lookup_contract_clause, resolve_item_key)}


class ToolRunner:
    """Dispatches registered tools and records an execution trace for the UI."""

    def __init__(self, agent: str):
        self.agent, self.traces = agent, []

    def call(self, name: str, **kwargs):
        t0 = time.perf_counter()
        result = TOOL_REGISTRY[name](**kwargs)
        self.traces.append(ToolTrace(agent=self.agent, tool=name, args=kwargs, result=result,
                                     ms=round((time.perf_counter() - t0) * 1000, 3)))
        return result

    def note(self, tool: str, args: dict, result):
        self.traces.append(ToolTrace(agent=self.agent, tool=tool, args=args, result=result))
