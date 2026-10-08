"""LangGraph pipeline: Ingestion -> Forensic Auditor -> Policy Critic -(reflect)-> Resolution."""
import hashlib
import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path
from typing import Any, Iterator, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from state import AuditState, Discrepancy, InvoiceDocument, LineItem, ToolTrace
from tools import ToolRunner, _q

MAX_REAUDIT = 1
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "25"))
LEDGER_PATH = Path(os.getenv("AUDIT_LEDGER_PATH", Path(__file__).parent / "data" / "audit_ledger.jsonl"))


def usd(x: float) -> str:
    return f"-${abs(x):,.2f}" if x < 0 else f"${x:,.2f}"


# =============================================================== LLM client
class LLMUnavailable(RuntimeError):
    pass


class LLMClient:
    """Thin wrapper over Groq / Gemini free tiers. provider='offline' disables all LLM calls."""

    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        env_g, env_m = os.getenv("GROQ_API_KEY"), os.getenv("GOOGLE_API_KEY")
        p = (provider or os.getenv("LLM_PROVIDER") or "auto").lower()
        if p == "auto":
            p = "groq" if (api_key or env_g) else "gemini" if env_m else "offline"
        self.provider = p
        self._failed = False  # circuit breaker: after one failure, skip the LLM for this audit
        self.api_key = api_key or (env_g if p == "groq" else env_m if p == "gemini" else None)
        self.model = (os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") if p == "groq"
                      else os.getenv("GEMINI_MODEL", "gemini-3.5-flash"))

    @property
    def available(self) -> bool:
        return self.provider in ("groq", "gemini") and bool(self.api_key) and not self._failed

    def _raw(self, system: str, user: str, json_mode: bool) -> str:
        if not self.available:
            raise LLMUnavailable("No LLM configured")
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(self._call, system, user, json_mode).result(timeout=LLM_TIMEOUT)
        except FutTimeout:
            self._failed = True
            raise TimeoutError(f"{self.provider} did not respond within {LLM_TIMEOUT:g}s") from None
        except Exception:
            self._failed = True
            raise
        finally:
            pool.shutdown(wait=False)

    def _call(self, system: str, user: str, json_mode: bool) -> str:
        if self.provider == "groq":
            from groq import Groq
            kw = {"response_format": {"type": "json_object"}} if json_mode else {}
            r = Groq(api_key=self.api_key).chat.completions.create(
                model=self.model, temperature=0, **kw,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            return r.choices[0].message.content
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        cfg = {"temperature": 0, **({"response_mime_type": "application/json"} if json_mode else {})}
        return genai.GenerativeModel(self.model, system_instruction=system).generate_content(
            user, generation_config=cfg).text

    def text(self, system: str, user: str) -> str:
        return self._raw(system, user, False).strip()

    def structured(self, system: str, user: str, model_cls, retries: int = 2):
        """JSON-mode extraction validated by Pydantic; validation errors are fed back on retry."""
        prompt = (f"{user}\n\nReturn ONLY a JSON object (no markdown) matching this JSON schema:\n"
                  f"{json.dumps(model_cls.model_json_schema())}")
        err = None
        for _ in range(retries + 1):
            raw = self._raw(system, prompt + (f"\n\nYour previous output failed validation: {err}" if err else ""), True)
            raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
            try:
                return model_cls.model_validate_json(raw)
            except ValidationError as e:
                err = str(e)[:600]
        raise ValueError(f"LLM output failed schema validation: {err}")


# ============================================================== 1. INGESTION
INGEST_SYSTEM = ("You are a meticulous invoice data-extraction engine. Copy values exactly as printed. "
                 "Never compute, correct or infer numbers. Dates as YYYY-MM-DD. tax_rate as a fraction.")


def _money(s: str) -> float:
    return float(s.replace(",", "").replace("$", ""))


def parse_invoice_regex(raw: str, vendor_id: str) -> InvoiceDocument:
    """Deterministic fallback parser for the fixture format (also used offline)."""
    def grab(pat, label, flags=re.M | re.I):
        m = re.search(pat, raw, flags)
        if not m:
            raise ValueError(f"Could not find {label} in invoice text")
        return m
    items = [LineItem(description=m["desc"].strip(), quantity=_money(m["qty"]),
                      unit_rate=_money(m["rate"]), line_total_claimed=_money(m["total"]))
             for m in re.finditer(r"^[-*•]?\s*(?P<desc>[^\n]+?):\s*(?P<qty>[\d.,]+)\s*(?:hours?|hrs?)\s*@\s*"
                                  r"\$(?P<rate>[\d.,]+)\s*/\s*(?:hr|hour)\s*=\s*\$(?P<total>[\d.,]+)", raw, re.M | re.I)]
    tax = grab(r"^Sales Tax\s*\((\d+(?:\.\d+)?)%\):\s*\$([\d,.]+)", "sales tax")
    disc = re.search(r"^[\w ]*Discount[^:\n]*:\s*-?\$([\d,.]+)", raw, re.M | re.I)
    return InvoiceDocument(
        vendor_name=re.sub(r"\s*\(.*\)\s*$", "", grab(r"^Vendor:\s*(.+)$", "vendor")[1]).strip(),
        vendor_id=vendor_id, invoice_number=grab(r"^Invoice\s*#:\s*(\S+)", "invoice number")[1],
        invoice_date=grab(r"^Invoice Date:\s*(\d{4}-\d{2}-\d{2})", "invoice date")[1],
        line_items=items, subtotal_claimed=_money(grab(r"^Subtotal:\s*\$([\d,.]+)", "subtotal")[1]),
        discount_claimed=_money(disc[1]) if disc else 0.0,
        tax_rate=float(tax[1]) / 100, tax_claimed=_money(tax[2]),
        total_claimed=_money(grab(r"^Total Due:\s*\$([\d,.]+)", "total")[1]))


def _ungrounded(inv: InvoiceDocument, raw: str) -> list[float]:
    """Anti-hallucination guard: every extracted figure must literally occur in the source text."""
    seen = {_money(m) for m in re.findall(r"\d[\d,]*\.?\d*", raw)}
    vals = [inv.subtotal_claimed, inv.tax_claimed, inv.total_claimed]
    vals += [v for li in inv.line_items for v in (li.quantity, li.unit_rate, li.line_total_claimed)]
    return [v for v in vals if v not in seen and round(v, 2) not in seen]


def make_ingestion(llm: LLMClient):
    def node(state: AuditState):
        run, raw, vid = ToolRunner("ingestion"), state["raw_invoice"], state["vendor_id"]
        inv, method = None, "regex_fallback"
        if llm.available:
            try:
                inv = llm.structured(INGEST_SYSTEM, f"Vendor ID: {vid}\n\nINVOICE TEXT:\n{raw}", InvoiceDocument)
                bad = _ungrounded(inv, raw)
                if bad:
                    run.note("grounding_check", {"ungrounded_values": bad}, "FAILED -> falling back to regex parser")
                    inv = None
                else:
                    method = f"llm:{llm.provider}/{llm.model}"
                    run.note("grounding_check", {}, "PASSED: all extracted figures appear in source text")
            except Exception as e:  # network, quota, schema...
                run.note("llm_extraction", {"provider": llm.provider}, f"ERROR {type(e).__name__}: {str(e)[:160]}")
        if inv is None:
            try:
                inv = parse_invoice_regex(raw, vid)
            except Exception as e:
                raise ValueError("Could not read this invoice. " + ("AI extraction is unavailable and "
                                 if not llm.available else "") + f"the built-in parser did not recognise the layout ({e}).") from e
        inv = inv.model_copy(update={"vendor_id": vid})  # caller-supplied vendor id is authoritative
        run.note("extract_invoice", {"method": method},
                 {"invoice_number": inv.invoice_number, "line_items": len(inv.line_items),
                  "total_claimed": inv.total_claimed})
        return {"invoice": inv, "ingestion_method": method, "traces": run.traces, "reaudit_count": 0}
    return node


# ======================================================= 2. FORENSIC AUDITOR
def _sub(a: float, b: float) -> float:
    return _q(a - b)


def make_auditor(llm: LLMClient):
    def llm_resolve(desc: str, keys: list[str]) -> Optional[str]:
        try:
            out = llm.text("Map an invoice line description to exactly one contract item key from the list, "
                           "or reply NONE. Reply with the key only.", f"Keys: {keys}\nDescription: {desc}")
            out = out.strip("`'\". \n").lower()
            return out if out in keys else None
        except Exception:
            return None

    def node(state: AuditState):
        inv, run = state["invoice"], ToolRunner("forensic_auditor")
        vid, reaudit = inv.vendor_id, state.get("reaudit_count", 0)
        findings: list[Discrepancy] = []

        def add(**kw):
            findings.append(Discrepancy(id=f"D{len(findings) + 1:02d}", **kw))

        resolved: list[Optional[tuple[str, dict]]] = []
        for i, li in enumerate(inv.line_items, 1):
            ref = f"Line {i}: {li.description}"
            calc = run.call("calc_line_item", qty=li.quantity, unit_rate=li.unit_rate)
            if (d := _sub(li.line_total_claimed, calc)) != 0:
                add(category="MATH_ERROR", line_ref=ref, billed=li.line_total_claimed, expected=calc, delta=d,
                    description=f"{li.quantity:g} hrs x {usd(li.unit_rate)} = {usd(calc)}, "
                                f"but invoice claims {usd(li.line_total_claimed)}")
            key = run.call("resolve_item_key", vendor_id=vid, description=li.description)["item_key"]
            if key is None and reaudit > 0 and llm.available:  # reflection pass: LLM-assisted mapping
                keys = run.call("lookup_contract_clause", vendor_id=vid, item_key="__list__").get("available_keys", [])
                key = llm_resolve(li.description, keys)
                run.note("llm_resolve_item_key", {"description": li.description}, key)
            if key is None:
                pol = run.call("lookup_contract_clause", vendor_id=vid, item_key="unscheduled_services")
                add(category="UNKNOWN_ITEM", line_ref=ref, billed=li.line_total_claimed, expected=0.0,
                    delta=li.line_total_claimed, clause=pol.get("clause"), clause_text=pol.get("clause_text"),
                    description=f"'{li.description}' is not in the contract rate schedule; cannot be verified")
                resolved.append(None)
                continue
            c = run.call("lookup_contract_clause", vendor_id=vid, item_key=key)
            resolved.append((key, c))
            if _sub(li.unit_rate, c["rate_usd_per_hr"]) > 0:
                at_contract = run.call("calc_line_item", qty=li.quantity, unit_rate=c["rate_usd_per_hr"])
                add(category="RATE_CREEP", line_ref=ref, billed=calc, expected=at_contract,
                    delta=_sub(calc, at_contract), clause=c["clause"], clause_text=c["clause_text"],
                    description=f"Billed {usd(li.unit_rate)}/hr vs contract {usd(c['rate_usd_per_hr'])}/hr "
                                f"(+{usd(_sub(li.unit_rate, c['rate_usd_per_hr']))}/hr) on {li.quantity:g} hrs")

        # hours caps (aggregated per role) + contract-correct line values
        used: dict[str, float] = {}
        exp_lines: list[float] = []
        roles: dict[str, dict] = {}
        for li, r in zip(inv.line_items, resolved):
            if r is None:
                exp_lines.append(li.line_total_claimed)  # unverifiable: carried at claimed value
                continue
            key, c = r
            roles[key] = c
            remaining = max(c["monthly_cap_hours"] - used.get(key, 0.0), 0.0)
            used[key] = used.get(key, 0.0) + li.quantity
            exp_lines.append(run.call("calc_line_item", qty=min(li.quantity, remaining), unit_rate=c["rate_usd_per_hr"]))
        for key, total in used.items():
            c = roles[key]
            if total > c["monthly_cap_hours"]:
                over = total - c["monthly_cap_hours"]
                amt = run.call("calc_line_item", qty=over, unit_rate=c["rate_usd_per_hr"])
                add(category="HOURS_CAP", line_ref=f"Role '{key}' (all lines)", delta=amt, clause=c["clause"],
                    clause_text=c["clause_text"],
                    billed=run.call("calc_line_item", qty=total, unit_rate=c["rate_usd_per_hr"]),
                    expected=run.call("calc_line_item", qty=c["monthly_cap_hours"], unit_rate=c["rate_usd_per_hr"]),
                    description=f"{total:g} hrs billed vs {c['monthly_cap_hours']:g} hr monthly cap: "
                                f"{over:g} hrs non-compliant (valued at contract rate)")

        # invoice-level checks
        disc_c = run.call("lookup_contract_clause", vendor_id=vid, item_key="volume_discount")
        tax_c = run.call("lookup_contract_clause", vendor_id=vid, item_key="tax")
        sum_claimed = run.call("calc_sum", values=[li.line_total_claimed for li in inv.line_items])
        if (d := _sub(inv.subtotal_claimed, sum_claimed)) != 0:
            add(category="SUBTOTAL_MISMATCH", line_ref="Subtotal", billed=inv.subtotal_claimed,
                expected=sum_claimed, delta=d, description="Subtotal differs from the sum of claimed line totals")
        exp_disc = run.call("calc_discount", subtotal=inv.subtotal_claimed,
                            threshold=disc_c["threshold_usd"], pct=disc_c["pct"])
        if (d := _sub(exp_disc, inv.discount_claimed)) != 0:
            add(category="DISCOUNT_ERROR", line_ref="Volume discount", billed=inv.discount_claimed,
                expected=exp_disc, delta=d, clause=disc_c["clause"], clause_text=disc_c["clause_text"],
                description=f"Discount claimed {usd(inv.discount_claimed)}, contract requires {usd(exp_disc)}")
        net = run.call("calc_total", subtotal=inv.subtotal_claimed, tax=0.0, discounts=inv.discount_claimed)
        exp_tax = run.call("calc_tax", subtotal=net, tax_rate=tax_c["rate"])
        if (d := _sub(inv.tax_claimed, exp_tax)) != 0:
            pre = run.call("calc_tax", subtotal=inv.subtotal_claimed, tax_rate=tax_c["rate"])
            if inv.discount_claimed > 0 and _sub(inv.tax_claimed, pre) == 0:
                why = (f"Tax charged on PRE-discount subtotal {usd(inv.subtotal_claimed)}; contract requires tax "
                       f"on post-discount amount {usd(net)}")
            elif abs(inv.tax_rate - tax_c["rate"]) > 1e-9:
                why = f"Tax rate {inv.tax_rate:.2%} differs from contract rate {tax_c['rate']:.2%}"
            else:
                why = "Tax arithmetic does not match contract basis"
            add(category="TAX_ERROR", line_ref="Sales tax", billed=inv.tax_claimed, expected=exp_tax, delta=d,
                clause=tax_c["clause"], clause_text=tax_c["clause_text"], description=why)
        exp_tot = run.call("calc_total", subtotal=inv.subtotal_claimed, tax=inv.tax_claimed, discounts=inv.discount_claimed)
        if (d := _sub(inv.total_claimed, exp_tot)) != 0:
            add(category="TOTAL_MISMATCH", line_ref="Total due", billed=inv.total_claimed, expected=exp_tot,
                delta=d, description="Total due differs from subtotal - discount + tax")

        # contract-correct reference invoice
        c_sub = run.call("calc_sum", values=exp_lines)
        c_disc = run.call("calc_discount", subtotal=c_sub, threshold=disc_c["threshold_usd"], pct=disc_c["pct"])
        c_net = run.call("calc_total", subtotal=c_sub, tax=0.0, discounts=c_disc)
        c_tax = run.call("calc_tax", subtotal=c_net, tax_rate=tax_c["rate"])
        c_tot = run.call("calc_total", subtotal=c_net, tax=c_tax)
        expected = {"subtotal": c_sub, "discount": c_disc, "tax": c_tax, "total": c_tot}
        return {"findings": findings, "expected": expected, "traces": run.traces, "needs_reaudit": False}
    return node


# ===================================================== 3. POLICY CRITIC
def make_critic(llm: LLMClient):
    def node(state: AuditState):
        run, vid = ToolRunner("policy_critic"), state["invoice"].vendor_id
        tol = run.call("lookup_contract_clause", vendor_id=vid, item_key="tolerance_usd")
        threshold, tclause = float(tol.get("value", 1.0)), tol.get("clause", "§8.1")
        count = state.get("reaudit_count", 0)
        unknown = [f for f in state["findings"] if f.category == "UNKNOWN_ITEM"]
        if unknown and count < MAX_REAUDIT:  # REFLECTION: send work back to the auditor
            run.note("reflect", {"unverifiable_lines": [f.line_ref for f in unknown]},
                     "Evidence incomplete -> requesting re-audit with LLM-assisted item mapping")
            return {"needs_reaudit": True, "reaudit_count": count + 1, "traces": run.traces}
        out = []
        for f in state["findings"]:
            if f.category == "UNKNOWN_ITEM":
                f = f.model_copy(update={"classification": "Needs Manual Review", "rationale":
                    "No contract rate exists after re-audit; cannot be approved or quantified automatically."})
            elif run.call("check_tolerance_threshold", delta=f.delta, threshold=threshold):
                f = f.model_copy(update={"classification": "Contractual Violation", "rationale":
                    f"Variance {usd(abs(f.delta))} exceeds the {usd(threshold)} tolerance ({tclause})."})
            else:
                f = f.model_copy(update={"classification": "Informational: Rounding Variance", "rationale":
                    f"Variance {usd(abs(f.delta))} is within the {usd(threshold)} tolerance ({tclause}); alert suppressed."})
            out.append(f)
        actionable = [f for f in out if f.classification != "Informational: Rounding Variance"]
        return {"findings": out, "actionable": actionable, "needs_reaudit": False,
                "suppressed": [f for f in out if f.classification == "Informational: Rounding Variance"],
                "verdict": "VIOLATIONS_FOUND" if actionable else "APPROVED_FOR_PAYMENT", "traces": run.traces}
    return node


def route_after_critic(state: AuditState) -> str:
    if state.get("needs_reaudit"):
        return "forensic_auditor"
    return "resolution_dispute" if state["verdict"] == "VIOLATIONS_FOUND" else "resolution_ledger"


# ================================================ 4. RESOLUTION / LEDGER
def _findings_table(rows: list[Discrepancy]) -> str:
    if not rows:
        return "_None_\n"
    out = "| ID | Type | Where | Billed | Contract-correct | Overcharge | Clause | Classification |\n|---|---|---|---|---|---|---|---|\n"
    for f in rows:
        out += (f"| {f.id} | {f.category} | {f.line_ref} | {usd(f.billed)} | {usd(f.expected)} | "
                f"{usd(f.delta)} | {f.clause or '—'} | {f.classification} |\n")
    return out


def build_report(state: AuditState) -> str:
    inv, exp = state["invoice"], state["expected"]
    calls: dict[str, int] = {}
    for t in state["traces"]:
        calls[t.tool] = calls.get(t.tool, 0) + 1
    over = _q(inv.total_claimed - exp["total"])
    md = (f"# Reconciliation Report — {inv.invoice_number}\n\n**Vendor:** {inv.vendor_name} ({inv.vendor_id})  \n"
          f"**Invoice date:** {inv.invoice_date}  \n**Verdict:** `{state['verdict']}`  \n"
          f"**Ingestion:** {state.get('ingestion_method')}  \n**Reflection passes:** {state.get('reaudit_count', 0)}\n\n"
          f"## Claimed vs contract-correct\n\n| | Claimed | Contract-correct |\n|---|---|---|\n"
          f"| Subtotal | {usd(inv.subtotal_claimed)} | {usd(exp['subtotal'])} |\n"
          f"| Discount | {usd(inv.discount_claimed)} | {usd(exp['discount'])} |\n"
          f"| Tax | {usd(inv.tax_claimed)} | {usd(exp['tax'])} |\n"
          f"| **Total** | **{usd(inv.total_claimed)}** | **{usd(exp['total'])}** |\n\n"
          f"**Net overbilling exposure:** {usd(over)} (includes cascading tax effects)\n\n"
          f"## Actionable findings\n\n{_findings_table(state['actionable'])}\n")
    for f in state["actionable"]:
        md += f"- **{f.id}** — {f.description}. {f.rationale}" + (f" Clause {f.clause}: _{f.clause_text}_" if f.clause_text else "") + "\n"
    md += f"\n## Suppressed (informational)\n\n{_findings_table(state['suppressed'])}\n"
    md += "## Tool execution summary\n\n" + ", ".join(f"`{k}` x{v}" for k, v in sorted(calls.items())) + "\n"
    return md


EMAIL_SYSTEM = ("You draft formal, courteous B2B billing-dispute emails as PLAIN TEXT for an email client. "
                "Never use Markdown: no asterisks, no bold, no headings, no backticks, no bullet symbols; "
                "number the items as '1.', '2.' on their own lines. Use ONLY the facts and dollar figures "
                "provided; do not compute or invent numbers. Cite every clause number exactly as given (e.g. §3.1). "
                "Do not quote clause wording at length; refer to it briefly. Sign off as 'Accounts Payable, Buyer Corp'. "
                "No placeholders in square brackets. Output the email only, starting with 'Subject:'.")


def _plain(text: str) -> str:
    """Strip any Markdown the model adds despite instructions."""
    text = re.sub(r"\*\*(.+?)\*\*|__(.+?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[*\-•]\s+", "", text, flags=re.M)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _email_template(state: AuditState) -> str:
    inv, exp = state["invoice"], state["expected"]
    lines = []
    for n, f in enumerate(state["actionable"], 1):
        cite = f"[Clause {f.clause}] " if f.clause else ""
        if f.classification == "Needs Manual Review":
            lines.append(f"{n}. {cite}{f.line_ref} — {usd(f.billed)}: {f.description}. Under {f.clause} this requires "
                         f"prior written approval; please provide the approval and supporting documentation.")
        else:
            lines.append(f"{n}. {cite}{f.line_ref} — {f.description}. Billed {usd(f.billed)}; contract-correct "
                         f"{usd(f.expected)}; difference {usd(f.delta)}.")
    credit = _q(inv.total_claimed - exp["total"])
    return (f"Subject: Dispute of Invoice {inv.invoice_number} — Billing Discrepancies under the MSA\n\n"
            f"Dear {inv.vendor_name} Accounts Receivable,\n\n"
            f"We have reconciled invoice {inv.invoice_number} dated {inv.invoice_date} (total claimed "
            f"{usd(inv.total_claimed)}) against our Master Services Agreement and identified the following "
            f"discrepancies:\n\n" + "\n".join(lines) + "\n\n"
            f"Based on the contract terms, the verifiable contract-correct total is {usd(exp['total'])}"
            f"{', a difference of ' + usd(credit) if credit > 0 else ''}. Please issue a corrected invoice or credit "
            f"memo addressing the items above. Payment of the disputed amount is on hold pending resolution; "
            f"undisputed amounts remain subject to the agreed payment terms.\n\n"
            f"Kind regards,\nAccounts Payable\nBuyer Corp\n\n[DRAFT — review before sending]")


def make_dispute(llm: LLMClient):
    def node(state: AuditState):
        run = ToolRunner("resolution")
        report = build_report(state)
        email, source = _email_template(state), "template"
        if llm.available:
            facts = [{"id": f.id, "clause": f.clause, "clause_text": f.clause_text, "where": f.line_ref,
                      "issue": f.description, "billed": usd(f.billed), "contract_correct": usd(f.expected),
                      "difference": usd(f.delta), "type": f.classification} for f in state["actionable"]]
            try:
                draft = llm.text(EMAIL_SYSTEM, json.dumps({"vendor": state["invoice"].vendor_name,
                    "invoice": state["invoice"].invoice_number, "findings": facts,
                    "contract_correct_total": usd(state["expected"]["total"]),
                    "claimed_total": usd(state["invoice"].total_claimed)}, indent=1))
                draft = _plain(draft)
                need = [f.clause for f in state["actionable"] if f.clause] + [usd(f.delta) for f in state["actionable"]
                                                                              if f.classification != "Needs Manual Review"]
                if all(tok in draft for tok in need):
                    email, source = draft, f"llm:{llm.provider}"
                else:
                    run.note("email_validation", {}, "LLM draft missing clause/amount -> using template")
            except Exception as e:
                run.note("email_llm", {}, f"ERROR {type(e).__name__}; using template")
        run.note("draft_dispute_email", {"source": source}, f"{len(email)} chars")
        return {"report_md": report, "dispute_email": email, "traces": run.traces}
    return node


def make_ledger(llm: LLMClient):
    def node(state: AuditState):
        run, inv = ToolRunner("resolution"), state["invoice"]
        terms = run.call("lookup_contract_clause", vendor_id=inv.vendor_id, item_key="payment_terms")
        days = int(re.search(r"\d+", str(terms.get("value", "30"))).group())
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rec = {"status": "APPROVED_FOR_PAYMENT", "invoice_number": inv.invoice_number, "vendor_id": inv.vendor_id,
               "vendor_name": inv.vendor_name, "invoice_date": str(inv.invoice_date),
               "amount_approved": inv.total_claimed, "payment_terms": terms.get("value"),
               "due_date": str(inv.invoice_date + timedelta(days=days)),
               "informational_variances_suppressed": len(state.get("suppressed", [])), "audited_at": stamp,
               "certificate_id": hashlib.sha256(f"{inv.invoice_number}{inv.total_claimed}{stamp}".encode()).hexdigest()[:12]}
        try:
            LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LEDGER_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            run.note("append_ledger", {"path": str(LEDGER_PATH)}, "written")
        except OSError as e:
            run.note("append_ledger", {"path": str(LEDGER_PATH)}, f"FAILED: {e}")
        cert = (f"# Reconciliation Certificate — {rec['certificate_id']}\n\n**{inv.invoice_number}** from "
                f"**{inv.vendor_name}** reconciles to the MSA.\n\n- Amount approved: **{usd(inv.total_claimed)}**\n"
                f"- Terms: {rec['payment_terms']} → due **{rec['due_date']}**\n"
                f"- Informational variances suppressed: {rec['informational_variances_suppressed']}\n\n"
                f"Status: `APPROVED_FOR_PAYMENT`\n\n" + "## Informational variances\n\n" + _findings_table(state.get("suppressed", [])))
        return {"ledger_record": rec, "report_md": cert, "traces": run.traces}
    return node


# ============================================================= graph wiring
def build_graph(llm: Optional[LLMClient] = None):
    llm = llm or LLMClient()
    g = StateGraph(AuditState)
    g.add_node("ingestion", make_ingestion(llm))
    g.add_node("forensic_auditor", make_auditor(llm))
    g.add_node("policy_critic", make_critic(llm))
    g.add_node("resolution_dispute", make_dispute(llm))   # Branch A
    g.add_node("resolution_ledger", make_ledger(llm))     # Branch B
    g.add_edge(START, "ingestion")
    g.add_edge("ingestion", "forensic_auditor")
    g.add_edge("forensic_auditor", "policy_critic")
    g.add_conditional_edges("policy_critic", route_after_critic,
                            {"forensic_auditor": "forensic_auditor",
                             "resolution_dispute": "resolution_dispute", "resolution_ledger": "resolution_ledger"})
    g.add_edge("resolution_dispute", END)
    g.add_edge("resolution_ledger", END)
    return g.compile()


def stream_audit(raw_invoice: str, vendor_id: str, llm: Optional[LLMClient] = None) -> Iterator[tuple[str, dict]]:
    """Yield (node_name, state_update) as each agent finishes."""
    graph = build_graph(llm)
    for chunk in graph.stream({"raw_invoice": raw_invoice, "vendor_id": vendor_id, "traces": []},
                              stream_mode="updates"):
        yield from chunk.items()


def merge_update(state: dict, update: dict) -> dict:
    for k, v in update.items():
        state[k] = state.get(k, []) + v if k == "traces" else v
    return state


def run_audit(raw_invoice: str, vendor_id: str, llm: Optional[LLMClient] = None) -> dict:
    state: dict = {"raw_invoice": raw_invoice, "vendor_id": vendor_id, "traces": []}
    for _, upd in stream_audit(raw_invoice, vendor_id, llm):
        merge_update(state, upd)
    return state
