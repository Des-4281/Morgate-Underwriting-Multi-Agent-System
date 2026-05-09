"""
Senior Mortgage Underwriting System - Local App Version

How to run:
1. Copy config.example.json to config.json.
2. Add your OpenAI API key to config.json.
3. Run: python app.py

Important:
- config.json is listed in .gitignore so your API key does not get committed.
- This app uses the OpenAI base URL: https://api.openai.com/v1
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Annotated

from openai import OpenAI


# ============================================================
# Config
# ============================================================

def load_config(path: str = "config.json") -> Dict[str, Any]:
    """Loads local config from config.json."""
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            "Missing config.json. Copy config.example.json to config.json "
            "and add your OpenAI API key."
        )

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME"]
    missing = [key for key in required_keys if not config.get(key)]

    if missing:
        raise ValueError(f"Missing required config keys: {', '.join(missing)}")

    return config


CONFIG = load_config()

client = OpenAI(
    api_key=CONFIG["OPENAI_API_KEY"],
    base_url=CONFIG["OPENAI_BASE_URL"],
)

MODEL_NAME = CONFIG["MODEL_NAME"]


# ============================================================
# Application State Schema
# ============================================================

class UnderwritingState(TypedDict):
    """The complete state of a loan application as it moves through the system."""

    # Application Information
    case_id: str
    applicant_data: Dict[str, Any]
    sanitized_data: Dict[str, Any]

    # Agent Analysis Results
    credit_analysis: Optional[str]
    income_analysis: Optional[str]
    asset_analysis: Optional[str]
    collateral_analysis: Optional[str]

    # Coordination & Decision
    critic_review: Optional[str]
    decision_memo: Optional[str]
    final_decision: Optional[str]  # APPROVED, DENIED, CONDITIONAL_APPROVAL
    risk_score: Optional[int]  # 0-100

    # Workflow Control
    next_agent: Optional[str]
    analysis_complete: bool
    human_review_required: bool
    human_review_completed: bool
    human_notes: Optional[str]

    # Compliance
    bias_flags: List[str]
    policy_violations: List[str]
    risk_flags: List[str]

    # Audit Trail
    reasoning_chain: Annotated[List[str], "append"]
    timestamp: str


# ============================================================
# Utility Helpers
# ============================================================

def call_llm(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
    """Calls the OpenAI chat completions endpoint and returns plain text."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or ""


def sum_current_monthly_debts(debts: Dict[str, Any]) -> float:
    """
    Sums only individual monthly debt line items.

    Excludes summary fields like total_monthly_debt so we do not double-count debt.
    """
    if not debts:
        return 0.0

    excluded_keys = {
        "total_monthly_debt",
        "total_debt",
        "monthly_debt_total",
    }

    return sum(
        amount
        for debt_name, amount in debts.items()
        if debt_name not in excluded_keys and isinstance(amount, (int, float))
    )


def calculate_dti_ratio(monthly_debt: float, monthly_income: float) -> str:
    """Calculates debt-to-income ratio."""
    if monthly_income <= 0:
        return "DTI cannot be calculated because monthly income is zero or negative."

    dti = (monthly_debt / monthly_income) * 100
    return f"DTI Ratio: {dti:.2f}% based on monthly debt ${monthly_debt:,.2f} and income ${monthly_income:,.2f}."


def calculate_ltv_ratio(loan_amount: float, property_value: float) -> str:
    """Calculates loan-to-value ratio."""
    if property_value <= 0:
        return "LTV cannot be calculated because property value is zero or negative."

    ltv = (loan_amount / property_value) * 100
    return f"LTV Ratio: {ltv:.2f}% based on loan amount ${loan_amount:,.2f} and property value ${property_value:,.2f}."


def calculate_reserves(liquid_assets: float, monthly_payment: float, required_months: int = 2) -> str:
    """Calculates available reserves in months."""
    if monthly_payment <= 0:
        return "Reserve months cannot be calculated because monthly payment is zero or negative."

    reserve_months = liquid_assets / monthly_payment
    status = "Meets requirement" if reserve_months >= required_months else "Does not meet requirement"

    return (
        f"Reserve Months: {reserve_months:.2f}. "
        f"Required: {required_months}. Status: {status}."
    )


def check_large_deposits(deposits: List[Dict[str, Any]], monthly_income: float) -> str:
    """Flags large deposits exceeding 50% of monthly income."""
    if not deposits:
        return "No large deposits provided."

    threshold = monthly_income * 0.50
    flagged = [
        deposit for deposit in deposits
        if isinstance(deposit.get("amount"), (int, float)) and deposit["amount"] >= threshold
    ]

    if not flagged:
        return "No large deposits exceed 50% of monthly income."

    details = "; ".join(
        f"${deposit.get('amount', 0):,.2f} from {deposit.get('source', 'unknown source')}"
        for deposit in flagged
    )

    return f"Large deposits requiring documentation: {details}."


def sanitize_applicant_data(applicant_data: Dict[str, Any]) -> Dict[str, Any]:
    """Masks direct personal identifiers before sending data to the LLM."""
    sanitized = dict(applicant_data)

    if "name" in sanitized:
        sanitized["name"] = "[APPLICANT_NAME]"

    if "ssn" in sanitized:
        ssn = str(sanitized["ssn"])
        sanitized["ssn"] = f"***-**-{ssn[-4:]}" if len(ssn) >= 4 else "***-**-XXXX"

    if "phone" in sanitized:
        phone = str(sanitized["phone"])
        sanitized["phone"] = f"***-***-{phone[-4:]}" if len(phone) >= 4 else "***-***-XXXX"

    if "email" in sanitized:
        sanitized["email"] = "[EMAIL_REDACTED]"

    return sanitized


def detect_bias_indicators(analysis: str) -> List[str]:
    """
    Flags potential protected-characteristic terms.

    Uses word boundaries and removes known underwriting-safe phrases like 'tradeline age'.
    """
    if not analysis:
        return []

    analysis_lower = analysis.lower()

    safe_phrases = [
        "tradeline age",
        "account age",
        "credit age",
        "age of account",
        "average age",
        "average age of accounts",
    ]

    analysis_for_bias = analysis_lower
    for phrase in safe_phrases:
        analysis_for_bias = analysis_for_bias.replace(phrase, "")

    protected_terms = [
        "age",
        "race",
        "religion",
        "gender",
        "sex",
        "marital status",
        "national origin",
        "disability",
        "family status",
        "familial status",
    ]

    flags = []
    for term in protected_terms:
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, analysis_for_bias):
            flags.append(f"Analysis mentions protected characteristic: {term}")

    return flags


def extract_risk_score(content: str) -> int:
    """Extracts RISK_SCORE from a structured memo."""
    match = re.search(r"RISK_SCORE:\s*(\d+)", content, re.IGNORECASE)
    if not match:
        return 50

    return max(0, min(100, int(match.group(1))))


def extract_decision(content: str) -> str:
    """Extracts DECISION from a structured memo."""
    decision_match = re.search(
        r"DECISION:\s*(APPROVED|DENIED|CONDITIONAL_APPROVAL|CONDITIONAL APPROVAL)",
        content,
        re.IGNORECASE,
    )

    if decision_match:
        return decision_match.group(1).upper().replace(" ", "_")

    return "CONDITIONAL_APPROVAL"


def extract_risk_flags(state: UnderwritingState, decision_content: str) -> List[str]:
    """Pulls practical risk flags from specialist analyses and final memo."""
    risk_keywords = [
        "below minimum",
        "high dti",
        "excessive dti",
        "late payment",
        "collection",
        "insufficient reserves",
        "large deposit",
        "short employment",
        "high ltv",
        "policy exception",
        "missing documentation",
        "condition required",
        "adverse",
        "denied",
    ]

    combined_analysis = "\n".join([
        state.get("credit_analysis", "") or "",
        state.get("income_analysis", "") or "",
        state.get("asset_analysis", "") or "",
        state.get("collateral_analysis", "") or "",
        state.get("critic_review", "") or "",
        decision_content or "",
    ])

    risk_flags: List[str] = []

    for line in combined_analysis.splitlines():
        clean_line = line.strip(" -•*0123456789.").strip()
        if clean_line and any(keyword in clean_line.lower() for keyword in risk_keywords):
            if clean_line not in risk_flags:
                risk_flags.append(clean_line)

    return risk_flags[:10]


# ============================================================
# State Initialization
# ============================================================

def initialize_application(applicant_data: Dict[str, Any]) -> UnderwritingState:
    """Creates the initial underwriting state."""
    case_id = applicant_data.get("case_id", f"CASE-{datetime.now().strftime('%Y%m%d%H%M%S')}")

    return {
        "case_id": case_id,
        "applicant_data": applicant_data,
        "sanitized_data": sanitize_applicant_data(applicant_data),

        "credit_analysis": None,
        "income_analysis": None,
        "asset_analysis": None,
        "collateral_analysis": None,

        "critic_review": None,
        "decision_memo": None,
        "final_decision": None,
        "risk_score": None,

        "next_agent": None,
        "analysis_complete": False,
        "human_review_required": False,
        "human_review_completed": False,
        "human_notes": None,

        "bias_flags": [],
        "policy_violations": [],
        "risk_flags": [],

        "reasoning_chain": [f"Application {case_id} initialized"],
        "timestamp": datetime.now().isoformat(),
    }


# ============================================================
# Specialist Agents
# ============================================================

def credit_analyst_node(state: UnderwritingState) -> UnderwritingState:
    app_data = state["sanitized_data"]
    credit = app_data.get("credit", {})

    system_prompt = """
You are a Senior Credit Analyst for mortgage underwriting.

Analyze the borrower's credit profile. Focus on:
1. Credit score strength
2. Payment history
3. Derogatory credit
4. Credit utilization
5. Depth and stability of credit history
6. Underwriting concerns and conditions

Do not use protected characteristics. Be objective and audit-ready.
"""

    user_prompt = f"""
Case ID: {state['case_id']}

Credit Data:
{json.dumps(credit, indent=2)}

Provide a concise credit analysis with strengths, risks, and recommended conditions.
"""

    content = call_llm(system_prompt, user_prompt)
    bias_flags = detect_bias_indicators(content)

    return {
        **state,
        "credit_analysis": content,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + ["Credit Analyst: completed credit review"],
    }


def income_analyst_node(state: UnderwritingState) -> UnderwritingState:
    app_data = state["sanitized_data"]
    employment = app_data.get("employment", {})
    debts = app_data.get("debts", {})
    loan = app_data.get("loan", {})

    monthly_income = float(employment.get("monthly_income", 0) or 0)
    proposed_payment = float(loan.get("estimated_payment", 0) or 0)
    total_debt = sum_current_monthly_debts(debts)

    dti_result = calculate_dti_ratio(
        monthly_debt=total_debt + proposed_payment,
        monthly_income=monthly_income,
    )

    housing_ratio_result = calculate_dti_ratio(
        monthly_debt=proposed_payment,
        monthly_income=monthly_income,
    ).replace("DTI Ratio", "Housing Ratio")

    debt_breakdown = {
        key: value
        for key, value in debts.items()
        if key not in {"total_monthly_debt", "total_debt", "monthly_debt_total"}
    }

    system_prompt = """
You are a Senior Income Analyst with expertise in mortgage underwriting.

Your task is to assess the borrower's income stability and capacity to repay.

Analysis framework:
1. Employment stability
2. Income verification
3. DTI ratio
4. Housing ratio
5. Debt obligations
6. Income-related risks
7. Conditions or recommendations

Important: Use the exact calculator results provided. Do not recalculate ratios manually.
"""

    user_prompt = f"""
Analyze income and repayment capacity for case {state['case_id']}.

Employment:
{json.dumps(employment, indent=2)}

Debt Breakdown:
{json.dumps(debt_breakdown, indent=2)}

Calculated DTI:
{dti_result}

Calculated Housing Ratio:
{housing_ratio_result}

Provide your income analysis based on these accurate calculations.
"""

    content = call_llm(system_prompt, user_prompt)
    bias_flags = detect_bias_indicators(content)

    return {
        **state,
        "income_analysis": content,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + ["Income Analyst: completed income review"],
    }


def asset_analyst_node(state: UnderwritingState) -> UnderwritingState:
    app_data = state["sanitized_data"]
    assets = app_data.get("assets", {})
    loan = app_data.get("loan", {})
    employment = app_data.get("employment", {})

    liquid_assets = float(assets.get("checking", 0) or 0) + float(assets.get("savings", 0) or 0)
    monthly_payment = float(loan.get("estimated_payment", 0) or 0)
    monthly_income = float(employment.get("monthly_income", 0) or 0)

    reserves_result = calculate_reserves(
        liquid_assets=liquid_assets,
        monthly_payment=monthly_payment,
        required_months=2,
    )

    deposits_result = check_large_deposits(
        deposits=assets.get("recent_deposits", []),
        monthly_income=monthly_income,
    )

    system_prompt = """
You are a Senior Asset Analyst with expertise in mortgage underwriting.

Your task is to assess assets, reserves, down payment funds, and documentation needs.

Use the provided reserve calculation and large deposit review. Do not recalculate manually.
"""

    user_prompt = f"""
Analyze assets and reserves for case {state['case_id']}.

Assets:
{json.dumps(assets, indent=2)}

Loan:
{json.dumps(loan, indent=2)}

Calculated Reserves:
{reserves_result}

Large Deposit Review:
{deposits_result}

Provide your asset analysis.
"""

    content = call_llm(system_prompt, user_prompt)
    bias_flags = detect_bias_indicators(content)

    return {
        **state,
        "asset_analysis": content,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + ["Asset Analyst: completed asset review"],
    }


def collateral_analyst_node(state: UnderwritingState) -> UnderwritingState:
    app_data = state["sanitized_data"]
    property_data = app_data.get("property", {})
    loan = app_data.get("loan", {})

    loan_amount = float(loan.get("loan_amount", 0) or 0)
    property_value = float(property_data.get("appraised_value", 0) or property_data.get("purchase_price", 0) or 0)

    ltv_result = calculate_ltv_ratio(
        loan_amount=loan_amount,
        property_value=property_value,
    )

    system_prompt = """
You are a Senior Collateral Analyst for mortgage underwriting.

Analyze property value, loan-to-value risk, occupancy, appraisal concerns, and collateral acceptability.
Use the provided LTV calculation. Do not recalculate manually.
"""

    user_prompt = f"""
Analyze collateral for case {state['case_id']}.

Property:
{json.dumps(property_data, indent=2)}

Loan:
{json.dumps(loan, indent=2)}

Calculated LTV:
{ltv_result}

Provide your collateral analysis.
"""

    content = call_llm(system_prompt, user_prompt)
    bias_flags = detect_bias_indicators(content)

    return {
        **state,
        "collateral_analysis": content,
        "bias_flags": state.get("bias_flags", []) + bias_flags,
        "reasoning_chain": state.get("reasoning_chain", []) + ["Collateral Analyst: completed collateral review"],
    }


def critic_agent_node(state: UnderwritingState) -> UnderwritingState:
    system_prompt = """
You are an underwriting quality-control critic.

Review all specialist analyses for:
1. Contradictions
2. Missing evidence
3. Policy concerns
4. Unclear recommendations
5. Fair-lending or bias concerns
6. Conditions that should be required

Be concise and practical.
"""

    user_prompt = f"""
Case ID: {state['case_id']}

Credit Analysis:
{state.get('credit_analysis')}

Income Analysis:
{state.get('income_analysis')}

Asset Analysis:
{state.get('asset_analysis')}

Collateral Analysis:
{state.get('collateral_analysis')}

Bias Flags:
{state.get('bias_flags')}

Provide a critic review.
"""

    content = call_llm(system_prompt, user_prompt)

    return {
        **state,
        "critic_review": content,
        "reasoning_chain": state.get("reasoning_chain", []) + ["Critic Agent: completed QC review"],
    }


def decision_agent_node(state: UnderwritingState) -> UnderwritingState:
    system_prompt = """
You are a Senior Mortgage Underwriting Decision Agent.

Your task is to synthesize all specialist analyses and produce a final underwriting recommendation.

Decision rules:
- APPROVED: Strong credit, income, assets, and collateral with low risk
- CONDITIONAL_APPROVAL: Generally acceptable but needs additional documentation or conditions
- DENIED: Material deficiencies, excessive risk, or failure to meet policy requirements

Required output format:
RISK_SCORE: [0-100]
DECISION: [APPROVED, DENIED, or CONDITIONAL_APPROVAL]
CREDIT_MEMO:
[Your memo]

Base the decision only on underwriting data and specialist analyses.
Do not use protected characteristics or biased reasoning.
"""

    user_prompt = f"""
Case ID: {state['case_id']}

Credit Analysis:
{state.get('credit_analysis')}

Income Analysis:
{state.get('income_analysis')}

Asset Analysis:
{state.get('asset_analysis')}

Collateral Analysis:
{state.get('collateral_analysis')}

Critic Review:
{state.get('critic_review')}

Bias Flags:
{state.get('bias_flags')}

Provide the final decision memo.
"""

    content = call_llm(system_prompt, user_prompt)

    risk_score = extract_risk_score(content)
    decision = extract_decision(content)
    risk_flags = extract_risk_flags(state, content)

    human_review_required = (
        risk_score >= 65
        or len(state.get("bias_flags", [])) > 0
        or decision == "DENIED"
    )

    return {
        **state,
        "decision_memo": content,
        "risk_score": risk_score,
        "final_decision": decision,
        "human_review_required": human_review_required,
        "risk_flags": risk_flags,
        "analysis_complete": True,
        "reasoning_chain": state.get("reasoning_chain", []) + [
            f"Decision Agent: final decision {decision} with risk score {risk_score}"
        ],
    }


# ============================================================
# Workflow Runner
# ============================================================

def run_underwriting(applicant_data: Dict[str, Any]) -> UnderwritingState:
    """Runs the full underwriting workflow."""
    state = initialize_application(applicant_data)
    state = credit_analyst_node(state)
    state = income_analyst_node(state)
    state = asset_analyst_node(state)
    state = collateral_analyst_node(state)
    state = critic_agent_node(state)
    state = decision_agent_node(state)
    return state


def load_test_cases(path: str = "mortgage_test_cases.json") -> List[Dict[str, Any]]:
    """Loads test cases from a local JSON file if present."""
    file_path = Path(path)

    if not file_path.exists():
        print(f"No {path} found. Using built-in sample case.")
        return [sample_case()]

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "test_cases" in data:
        return data["test_cases"]

    if isinstance(data, list):
        return data

    raise ValueError("Unsupported test case JSON format. Expected list or {'test_cases': [...]}.")


def sample_case() -> Dict[str, Any]:
    """Built-in sample case for a quick local smoke test."""
    return {
        "case_id": "TC-SAMPLE-01",
        "name": "Sample Applicant",
        "email": "sample@example.com",
        "phone": "555-555-1212",
        "credit": {
            "credit_score": 720,
            "late_payments": 0,
            "collections": 0,
            "utilization": 0.28,
            "tradeline_age_years": 6,
        },
        "employment": {
            "employer": "Example Company",
            "position": "Operations Manager",
            "type": "W2",
            "years_employed": 4,
            "monthly_income": 8500,
        },
        "debts": {
            "credit_card": 350,
            "auto_loan": 500,
            "student_loan": 250,
            "total_monthly_debt": 1100,
        },
        "assets": {
            "checking": 9000,
            "savings": 18000,
            "recent_deposits": [
                {"amount": 1200, "source": "payroll"},
                {"amount": 7000, "source": "gift"},
            ],
        },
        "loan": {
            "loan_amount": 360000,
            "down_payment": 40000,
            "estimated_payment": 3650,
        },
        "property": {
            "purchase_price": 400000,
            "appraised_value": 405000,
            "occupancy": "primary residence",
            "property_type": "single family",
        },
    }


def print_report(state: UnderwritingState) -> None:
    """Prints a simple terminal report."""
    print("\n" + "=" * 80)
    print(f"UNDERWRITING REPORT - {state['case_id']}")
    print("=" * 80)

    print(f"\nFinal Decision: {state.get('final_decision')}")
    print(f"Risk Score: {state.get('risk_score')}")
    print(f"Human Review Required: {state.get('human_review_required')}")

    print("\nRisk Flags:")
    if state.get("risk_flags"):
        for flag in state["risk_flags"]:
            print(f"- {flag}")
    else:
        print("- None identified")

    print("\nBias Flags:")
    if state.get("bias_flags"):
        for flag in state["bias_flags"]:
            print(f"- {flag}")
    else:
        print("- None identified")

    print("\nDecision Memo:")
    print(state.get("decision_memo") or "No decision memo generated.")

    print("\nReasoning Chain:")
    for step in state.get("reasoning_chain", []):
        print(f"- {step}")


def save_report(state: UnderwritingState, output_dir: str = "outputs") -> Path:
    """Saves the final state as a JSON report."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    report_path = output_path / f"{state['case_id']}_underwriting_report.json"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)

    return report_path


def main() -> None:
    test_cases = load_test_cases()

    for applicant_data in test_cases:
        state = run_underwriting(applicant_data)
        print_report(state)
        report_path = save_report(state)
        print(f"\nSaved report to: {report_path}")


if __name__ == "__main__":
    main()
