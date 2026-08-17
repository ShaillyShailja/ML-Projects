"""
JE Mapping Prototype — Rule-Based + LLM (Ollama) Hybrid Pipeline
==================================================================

Purpose
-------
Demonstrates the end-to-end architecture for the audit journal-entry
mapping tool. Natural-language conditional logic is interpreted by a
locally running Ollama model (no data leaves the machine — Ollama
serves the model on localhost). If Ollama isn't running or reachable,
the pipeline automatically falls back to MockLLM so the demo never
breaks.

IMPORTANT — before running this anywhere near a Deloitte laptop or
real client data: confirm with Deloitte IT/security/GenAI governance
that local LLM tooling like Ollama is approved for use. This script
is built and intended for use with SYNTHETIC data only, on a personal
machine, until that's confirmed. Nothing about "runs locally" implies
"automatically approved."

Setup (one-time, on your own machine)
--------------------------------------
1. Install Ollama:            https://ollama.com/download
2. Pull a small instruct model, e.g.:
       ollama pull llama3.1:8b
   (or `phi3`, `mistral`, `qwen2.5:7b` — any instruct-tuned model works;
   smaller models are faster but may need prompt tweaks for reliability)
3. Ollama runs a local server automatically at http://localhost:11434
4. pip install requests

Pipeline
--------
Data Request Form (DRF)
    -> Parser (reads rows)
    -> Mapping Classifier (direct / transformation / conditional)
        -> Direct / Transformation -> Rule Engine
        -> Conditional (natural language) -> LLM (Ollama, local)
    -> Structured mapping spec (JSON)
    -> Validation layer
    -> Auditor review (Accept / Edit / Reject) — simulated here
    -> Execution engine (pandas stands in for PySpark in this demo)
    -> Mapped output
"""

from __future__ import annotations
import re
import json
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# 1. SYNTHETIC DATA REQUEST FORM (DRF)
# ---------------------------------------------------------------------------
# In production this comes from the Excel/CSV the audit team fills in.
# "standard_field" = target schema field. "client_logic" = free-text
# instruction from the auditor describing how to derive it from the
# client's raw GL columns.

SYNTHETIC_DRF = [
    {"standard_field": "journal_id", "client_logic": "Directly map JE_ID"},
    {"standard_field": "posting_date", "client_logic": "Directly map POST_DATE"},
    {"standard_field": "entity", "client_logic": "Directly map COMPANY_CODE"},
    {"standard_field": "amount", "client_logic": "DEBIT_AMOUNT - CREDIT_AMOUNT"},
    {
        "standard_field": "classification",
        "client_logic": (
            "If the account number starts with 4, classify the transaction "
            "as Revenue. Otherwise classify it as Expense."
        ),
    },
    {
        "standard_field": "risk_flag",
        "client_logic": (
            "If amount is greater than 1000000 and document type is 'AB', "
            "flag as High risk. Otherwise flag as Low risk."
        ),
    },
]

# Synthetic raw GL data the mapping will be executed against.
SYNTHETIC_GL_DATA = pd.DataFrame(
    [
        {"JE_ID": "JE001", "POST_DATE": "2026-01-05", "COMPANY_CODE": "1000",
         "DEBIT_AMOUNT": 1_200_000, "CREDIT_AMOUNT": 0, "ACCOUNT": "4001",
         "DOCUMENT_TYPE": "AB"},
        {"JE_ID": "JE002", "POST_DATE": "2026-01-06", "COMPANY_CODE": "1000",
         "DEBIT_AMOUNT": 0, "CREDIT_AMOUNT": 50_000, "ACCOUNT": "5002",
         "DOCUMENT_TYPE": "SA"},
        {"JE_ID": "JE003", "POST_DATE": "2026-01-07", "COMPANY_CODE": "2000",
         "DEBIT_AMOUNT": 2_000_000, "CREDIT_AMOUNT": 0, "ACCOUNT": "4010",
         "DOCUMENT_TYPE": "AB"},
    ]
)


# ---------------------------------------------------------------------------
# 2. MAPPING SPEC — the structured JSON every path converges on
# ---------------------------------------------------------------------------

@dataclass
class MappingSpec:
    target_field: str
    type: str  # "direct" | "transformation" | "conditional"
    source_column: Optional[str] = None
    expression: Optional[str] = None
    conditions: list[dict] = field(default_factory=list)
    if_true: Optional[Any] = None
    if_false: Optional[Any] = None
    raw_logic: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# 3. MAPPING CLASSIFIER — decides direct vs. transformation vs. conditional
# ---------------------------------------------------------------------------

class MappingClassifier:
    """Cheap, deterministic pre-check before anything touches the LLM.

    This is the piece that keeps LLM usage scoped to only the cases
    that actually need natural-language understanding.
    """

    DIRECT_PATTERN = re.compile(r"^directly map (\w+)$", re.IGNORECASE)
    TRANSFORM_PATTERN = re.compile(r"^[\w\s]+[-+*/][\w\s]+$")

    @classmethod
    def classify(cls, logic_text: str) -> str:
        text = logic_text.strip()
        if cls.DIRECT_PATTERN.match(text):
            return "direct"
        if cls.TRANSFORM_PATTERN.match(text) and "if" not in text.lower():
            return "transformation"
        return "conditional"


# ---------------------------------------------------------------------------
# 4. RULE ENGINE — handles "direct" and "transformation" without any LLM
# ---------------------------------------------------------------------------

class RuleEngine:
    @staticmethod
    def build_direct(standard_field: str, logic_text: str) -> MappingSpec:
        match = MappingClassifier.DIRECT_PATTERN.match(logic_text.strip())
        source_col = match.group(1)
        return MappingSpec(
            target_field=standard_field,
            type="direct",
            source_column=source_col,
            raw_logic=logic_text,
        )

    @staticmethod
    def build_transformation(standard_field: str, logic_text: str) -> MappingSpec:
        return MappingSpec(
            target_field=standard_field,
            type="transformation",
            expression=logic_text.strip(),
            raw_logic=logic_text,
        )


# ---------------------------------------------------------------------------
# 5. MOCK LLM — stands in for a real model call
# ---------------------------------------------------------------------------
# Real version would send `logic_text` as a prompt to an approved model
# and parse its JSON response. This mock uses simple pattern extraction
# so the rest of the pipeline can be built and tested today.

class MockLLM:
    """Swap `interpret()` for a real API/local-model call later.
    Everything downstream only depends on this method's return shape.
    """

    COND_STARTS_WITH = re.compile(
        r"if (?:the )?(\w+) (?:number )?starts with ['\"]?(\w+)['\"]?,? "
        r"(?:classify (?:the transaction )?as|flag as) (\w+)\.? "
        r"otherwise,? (?:classify (?:it )?as|flag as) (\w+)",
        re.IGNORECASE,
    )
    COND_AND_THRESHOLD = re.compile(
        r"if (\w+) is greater than (\d+) and ([\w ]+?) is ['\"]?(\w+)['\"]?,? "
        r"flag as (\w+) risk\.? otherwise,? flag as (\w+) risk",
        re.IGNORECASE,
    )

    def interpret(self, standard_field: str, logic_text: str) -> MappingSpec:
        text = logic_text.strip()

        m = self.COND_STARTS_WITH.search(text)
        if m:
            column, prefix, if_true, if_false = m.groups()
            return MappingSpec(
                target_field=standard_field,
                type="conditional",
                conditions=[{"column": column.upper(), "operator": "starts_with", "value": prefix}],
                if_true=if_true,
                if_false=if_false,
                raw_logic=logic_text,
            )

        m = self.COND_AND_THRESHOLD.search(text)
        if m:
            amt_field, threshold, doc_field, doc_val, if_true, if_false = m.groups()
            return MappingSpec(
                target_field=standard_field,
                type="conditional",
                conditions=[
                    {"column": amt_field.upper(), "operator": ">", "value": int(threshold)},
                    {"column": doc_field.strip().replace(" ", "_").upper(), "operator": "==", "value": doc_val},
                ],
                if_true=f"{if_true} risk",
                if_false=f"{if_false} risk",
                raw_logic=logic_text,
            )

        # Fallback: real LLM would still return something; mock flags for manual review.
        return MappingSpec(
            target_field=standard_field,
            type="conditional",
            conditions=[],
            raw_logic=logic_text,
        )


# ---------------------------------------------------------------------------
# 5b. OLLAMA LLM — real local model, same interface as MockLLM
# ---------------------------------------------------------------------------

OLLAMA_SYSTEM_PROMPT_TEMPLATE = """You convert an auditor's plain-English field-mapping \
instruction into a structured JSON mapping specification. Respond with ONLY \
valid JSON — no explanation, no markdown fences, no code blocks.

The GL data has EXACTLY these columns available — you MUST use one of these \
exact names (case-sensitive) for every "column" value. Do NOT invent, \
abbreviate, or guess a different name:
{column_list}

Schema:
{{
  "type": "conditional",
  "conditions": [
    {{"column": "<ONE OF THE EXACT COLUMN NAMES ABOVE>", "operator": "starts_with|==|>|<|>=|<=", "value": <string or number>}}
  ],
  "if_true": "<value or label to assign>",
  "if_false": "<value or label to assign>"
}}

Rules:
- "column" must be copied exactly from the list above — never modify, shorten, or guess it.
- If multiple conditions are joined with "and", include multiple entries in "conditions".
- If the instruction references a field that has no match in the column list \
(e.g. a derived field computed in an earlier mapping step, like "amount"), \
still pick the closest real source column available, or if truly ambiguous, return:
  {{"type": "conditional", "conditions": [], "if_true": null, "if_false": null}}

Example
-------
Columns available: ACCOUNT, DOCUMENT_TYPE, DEBIT_AMOUNT, CREDIT_AMOUNT, COMPANY_CODE

Instruction: "If the account number starts with 4, classify the transaction as \
Revenue. Otherwise classify it as Expense."

Output:
{{"type": "conditional", "conditions": [{{"column": "ACCOUNT", "operator": "starts_with", "value": "4"}}], "if_true": "Revenue", "if_false": "Expense"}}
"""


class ModelNotPulledError(Exception):
    """Raised when Ollama is running but the requested model tag hasn't
    been pulled locally yet (`ollama pull <model>`)."""
    pass


class OllamaLLM:
    """Calls a locally running Ollama model. Same interface as MockLLM —
    `interpret(standard_field, logic_text) -> MappingSpec` — so it's a
    drop-in replacement anywhere MockLLM is used.

    Requires Ollama running locally (default: http://localhost:11434)
    with a model already pulled, e.g.:  ollama pull llama3.1:8b
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        host: str = "http://localhost:11434",
        timeout: int = 30,
        available_columns: Optional[list[str]] = None,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        # Real schema, injected into the prompt so the model can't invent names.
        self.available_columns = available_columns or []

    def is_available(self) -> bool:
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def get_installed_models(self) -> list[str]:
        """List of model tags Ollama actually has pulled locally."""
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=3)
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except requests.exceptions.RequestException:
            return []

    def model_is_pulled(self) -> bool:
        installed = self.get_installed_models()
        # Ollama tags include a version suffix (e.g. "llama3.1:8b"); match on
        # either the exact tag or the base name before ":" to be lenient.
        base = self.model.split(":")[0]
        return any(m == self.model or m.split(":")[0] == base for m in installed)

    def _build_system_prompt(self) -> str:
        column_list = "\n".join(f"- {c}" for c in self.available_columns) or "- (none provided)"
        return OLLAMA_SYSTEM_PROMPT_TEMPLATE.format(column_list=column_list)

    def _closest_column(self, name: str) -> str:
        """Fuzzy-corrects near-miss column names (e.g. 'DOC_TYPE' -> 'DOCUMENT_TYPE')
        so small model drift doesn't unnecessarily fail validation.
        """
        if not name or not self.available_columns:
            return name
        if name in self.available_columns:
            return name
        import difflib
        match = difflib.get_close_matches(name, self.available_columns, n=1, cutoff=0.5)
        return match[0] if match else name

    def interpret(self, standard_field: str, logic_text: str) -> MappingSpec:
        payload = {
            "model": self.model,
            "system": self._build_system_prompt(),
            "prompt": f"Instruction:\n{logic_text.strip()}",
            "stream": False,
            "format": "json",   # Ollama enforces valid JSON output when supported
            "options": {"temperature": 0},
        }

        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            if resp.status_code == 404:
                # Ollama returns 404 on this endpoint when the model tag isn't
                # pulled locally, NOT when the server is down (is_available()
                # already confirmed the server responds). Surface that clearly.
                try:
                    server_msg = resp.json().get("error", resp.text)
                except ValueError:
                    server_msg = resp.text
                installed = self.get_installed_models()
                raise ModelNotPulledError(
                    f"Model '{self.model}' isn't available on the Ollama server "
                    f"({self.host}). Server said: {server_msg!r}. "
                    f"Installed models: {installed or '(none)'}. "
                    f"Fix: run `ollama pull {self.model}` in a terminal."
                )
            resp.raise_for_status()
            raw_text = resp.json().get("response", "").strip()
            parsed = json.loads(raw_text)
        except (requests.exceptions.RequestException, json.JSONDecodeError, KeyError, ModelNotPulledError) as e:
            # Fail safe: return an empty conditional spec so the validation
            # layer routes it to "needs manual mapping" instead of crashing.
            print(f"[LLM] Error interpreting '{standard_field}': {e}")
            return MappingSpec(
                target_field=standard_field,
                type="conditional",
                conditions=[],
                raw_logic=logic_text,
                if_true=None,
                if_false=f"__LLM_ERROR__: {e}",
            )

        conditions = parsed.get("conditions", []) or []
        for cond in conditions:
            if "column" in cond:
                corrected = self._closest_column(cond["column"])
                if corrected != cond["column"]:
                    cond["_original_column"] = cond["column"]  # kept for debugging/audit trail
                cond["column"] = corrected

        return MappingSpec(
            target_field=standard_field,
            type="conditional",
            conditions=conditions,
            if_true=parsed.get("if_true"),
            if_false=parsed.get("if_false"),
            raw_logic=logic_text,
        )


def get_llm(
    prefer_ollama: bool = True,
    model: str = "llama3.1:8b",
    available_columns: Optional[list[str]] = None,
):
    """Returns a live OllamaLLM if reachable AND the model is actually
    pulled, otherwise falls back to MockLLM automatically — so the demo
    never breaks if Ollama isn't running or the model isn't downloaded.
    """
    if prefer_ollama:
        ollama = OllamaLLM(model=model, available_columns=available_columns)
        if not ollama.is_available():
            print(f"[LLM] Ollama server not reachable at {ollama.host} — falling back to MockLLM.")
            return MockLLM()
        if not ollama.model_is_pulled():
            installed = ollama.get_installed_models()
            print(
                f"[LLM] Ollama is running, but model '{model}' isn't pulled yet.\n"
                f"      Installed models: {installed or '(none)'}\n"
                f"      Fix: run `ollama pull {model}` in a terminal, then re-run.\n"
                f"      Falling back to MockLLM for now."
            )
            return MockLLM()
        print(f"[LLM] Using Ollama ({model}) at {ollama.host}")
        return ollama
    return MockLLM()


# ---------------------------------------------------------------------------
# 6. VALIDATION LAYER
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    pass


class Validator:
    VALID_OPERATORS = {"starts_with", "==", ">", "<", ">=", "<="}

    def __init__(self, available_columns: list[str]):
        self.available_columns = set(available_columns)

    def validate(self, spec: MappingSpec) -> list[str]:
        """Returns a list of warnings/errors. Empty list = clean."""
        issues = []

        if spec.type == "direct":
            if spec.source_column not in self.available_columns:
                issues.append(f"Source column '{spec.source_column}' not found in GL data.")

        elif spec.type == "transformation":
            referenced = re.findall(r"[A-Z_]+", spec.expression or "")
            for col in referenced:
                if col not in self.available_columns:
                    issues.append(f"Expression references unknown column '{col}'.")

        elif spec.type == "conditional":
            if not spec.conditions:
                issues.append(
                    "Could not parse conditional logic automatically — "
                    "needs manual mapping via UI."
                )
            for cond in spec.conditions:
                if cond["column"] not in self.available_columns:
                    issues.append(f"Condition references unknown column '{cond['column']}'.")
                if cond["operator"] not in self.VALID_OPERATORS:
                    issues.append(f"Unsupported operator '{cond['operator']}'.")

        return issues


# ---------------------------------------------------------------------------
# 7. EXECUTION ENGINE (pandas here; PySpark in production)
# ---------------------------------------------------------------------------

class Executor:
    @staticmethod
    def _coerce_value(series: pd.Series, val: Any) -> Any:
        """Coerces `val` (which may arrive as a string from the LLM's JSON,
        e.g. "1000000" instead of 1000000) to match the target column's dtype,
        so numeric comparisons don't fail on a type mismatch.
        """
        if pd.api.types.is_numeric_dtype(series) and isinstance(val, str):
            try:
                return float(val) if "." in val else int(val)
            except ValueError:
                return val  # not actually numeric-looking; let it fail downstream loudly
        return val

    @staticmethod
    def apply(spec: MappingSpec, df: pd.DataFrame) -> pd.Series:
        if spec.type == "direct":
            return df[spec.source_column]

        if spec.type == "transformation":
            expr = spec.expression
            for col in df.columns:
                expr = re.sub(rf"\b{col}\b", f"df['{col}']", expr)
            return eval(expr)  # demo only — production uses a safe expression evaluator

        if spec.type == "conditional":
            mask = pd.Series([True] * len(df))
            for cond in spec.conditions:
                col, op, val = cond["column"], cond["operator"], cond["value"]
                val = Executor._coerce_value(df[col], val)
                if op == "starts_with":
                    mask &= df[col].astype(str).str.startswith(str(val))
                elif op == "==":
                    mask &= df[col].astype(str) == str(val)
                elif op == ">":
                    mask &= df[col] > val
                elif op == "<":
                    mask &= df[col] < val
                elif op == ">=":
                    mask &= df[col] >= val
                elif op == "<=":
                    mask &= df[col] <= val
            return mask.map({True: spec.if_true, False: spec.if_false})

        raise ValueError(f"Unknown mapping type: {spec.type}")


# ---------------------------------------------------------------------------
# 8. PIPELINE — ties it all together
# ---------------------------------------------------------------------------

def run_pipeline(
    drf: list[dict],
    gl_data: pd.DataFrame,
    use_ollama: bool = True,
    ollama_model: str = "llama3.1:8b",
) -> tuple[pd.DataFrame, list[dict]]:
    llm = get_llm(
        prefer_ollama=use_ollama,
        model=ollama_model,
        available_columns=list(gl_data.columns),
    )
    validator = Validator(available_columns=list(gl_data.columns))

    specs = []
    review_log = []

    for row in drf:
        field_name = row["standard_field"]
        logic_text = row["client_logic"]
        mapping_type = MappingClassifier.classify(logic_text)

        if mapping_type == "direct":
            spec = RuleEngine.build_direct(field_name, logic_text)
        elif mapping_type == "transformation":
            spec = RuleEngine.build_transformation(field_name, logic_text)
        else:
            spec = llm.interpret(field_name, logic_text)

        issues = validator.validate(spec)
        specs.append(spec)
        review_log.append({
            "target_field": field_name,
            "type": spec.type,
            "generated_spec": spec.to_dict(),
            "validation_issues": issues,
            "auditor_decision": "Accept" if not issues else "Needs Review",
        })

    # Simulated "auditor accepted everything with no issues" for the demo run.
    result_df = gl_data.copy()
    output = pd.DataFrame(index=gl_data.index)
    for spec, log in zip(specs, review_log):
        if not log["validation_issues"]:
            output[spec.target_field] = Executor.apply(spec, result_df)
        else:
            output[spec.target_field] = "NEEDS_MANUAL_MAPPING"

    return output, review_log


# ---------------------------------------------------------------------------
# 9. DEMO
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Set use_ollama=False to force MockLLM (e.g. no Ollama installed yet).
    mapped_output, log = run_pipeline(
        SYNTHETIC_DRF, SYNTHETIC_GL_DATA, use_ollama=True, ollama_model="llama3.1:8b"
    )

    print("=" * 70)
    print("MAPPING SPECS GENERATED (this is what the auditor UI would show)")
    print("=" * 70)
    print(json.dumps(log, indent=2, default=str))
    
    print("\n" + "=" * 70)
    print("INPUT CDRF")
    print("=" * 70)
    print(SYNTHETIC_DRF)

    
    print("\n" + "=" * 70)
    print("INPUT GL")
    print("=" * 70)
    print(SYNTHETIC_GL_DATA.to_string(index=False))

    print("\n" + "=" * 70)
    print("MAPPED OUTPUT (standard schema)")
    print("=" * 70)
    print(mapped_output.to_string(index=False))