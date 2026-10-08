# Ledgerwise

Multi-agent invoice audit. Upload a vendor invoice (PDF or text) and Ledgerwise checks it against the vendor's
contract with deterministic tools, filters rounding noise, then drafts a dispute email or approves the invoice.

## Run
```powershell
pip install -r requirements.txt
python server.py          # opens http://localhost:8000
```
Configuration lives in `.env` (loaded automatically): `GOOGLE_API_KEY`, `LLM_PROVIDER`, `GEMINI_MODEL`.
Copy `.env.example` to `.env` to start. Without a key the app falls back to a built-in parser.

## Layout
```
ledgerwise/
├── server.py        web server and API
├── graph.py         LangGraph agents: ingestion, auditor, critic, resolution
├── tools.py         deterministic arithmetic and contract lookups
├── state.py         Pydantic schemas and shared state
├── web/index.html   dashboard
├── test_data/       contract_vendor_a.md and four sample invoices
├── test_pipeline.py acceptance tests (offline): pytest -q
└── data/            audit_ledger.jsonl, created at runtime
```
The vendor is detected from the invoice (`Vendor: Name (VENDOR-A)`), and its contract is read from
`test_data/contract_<vendor>.md`. LLM calls time out after 25 s (`LLM_TIMEOUT`) and fall back automatically.
