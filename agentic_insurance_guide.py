import os
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI
from search_guidelines_qdrant import QdrantGuidelineSearcher
from dotenv import load_dotenv

load_dotenv()
# ----------------- OPENAI / AZURE CONFIG -----------------

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not (AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY and AZURE_OPENAI_DEPLOYMENT):
    raise RuntimeError(
        "Missing one of AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_API_KEY / AZURE_OPENAI_CHAT_DEPLOYMENT"
    )

client = OpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    base_url=f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_DEPLOYMENT}",
    default_query={"api-version": AZURE_OPENAI_API_VERSION},
)

searcher = QdrantGuidelineSearcher()


# ----------------- TOOL IMPLEMENTATIONS (Python side) -----------------

def route_classifier_tool_impl(question: str) -> Dict[str, Any]:
    """
    Decide insurance department + intent_type based on the question.

    department examples:
      - 'Underwriting'
      - 'Claims'
      - 'Fraud Control'
      - 'Customer Service'
      - 'Life Claims'
      - 'General Insurance'

    intent_type examples:
      - 'guideline'
      - 'workflow'
      - 'premium_calculation'
      - 'document_requirements'
    """
    q = question.lower()
    department = "General Insurance"
    intent_type = "guideline"
    notes: List[str] = []

    # Underwriting / health
    if any(w in q for w in ["underwriting", "health policy", "pre-existing", "medical test"]):
        department = "Underwriting"
        intent_type = "guideline"
        notes.append("Underwriting / health policy context detected.")

    # Motor / vehicle claims
    if any(w in q for w in ["motor", "vehicle", "garage", "survey", "accident claim", "own damage"]):
        department = "Claims"
        notes.append("Motor / claims context detected.")

    # Fraud control
    if any(w in q for w in ["fraud", "red flag", "suspicious", "investigation"]):
        department = "Fraud Control"
        intent_type = "guideline"
        notes.append("Fraud-related context detected.")

    # Renewal / lapse / reinstatement
    if any(w in q for w in ["renewal", "lapse", "reinstatement", "grace period"]):
        department = "Customer Service"
        intent_type = "workflow"
        notes.append("Policy servicing / renewal context detected.")

    # Life claims
    if any(w in q for w in ["life claim", "death certificate", "nominee", "accidental death"]):
        department = "Life Claims"
        notes.append("Life insurance claim context detected.")

    # Premium calculation intent
    if any(w in q for w in ["premium", "calculate", "pricing", "rate", "loading", "discount"]):
        intent_type = "premium_calculation"
        notes.append("Premium calculation intent detected.")

    # Workflow / steps
    if any(w in q for w in ["step-by-step", "steps", "process", "workflow", "procedure"]):
        intent_type = "workflow"
        notes.append("User asked for steps / workflow.")

    # Document requirements
    if any(w in q for w in ["documents required", "required documents", "documentation", "what documents"]):
        intent_type = "document_requirements"
        notes.append("Document requirement intent detected.")

    return {
        "department": department,
        "intent_type": intent_type,
        "notes": notes,
    }


def search_guidelines_tool_impl(
    query: str,
    department: Optional[str] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Call QdrantGuidelineSearcher and return INSURANCE guideline sections.
    """
    filters: Dict[str, str] = {}
    if department:
        filters["department"] = department

    results = searcher.search_guidelines(
        query=query,
        filters=filters,
        top_k=top_k,
        score_threshold=0.3,
    )

    return {
        "query": query,
        "department": department,
        "results": results,
    }


def summarize_chunks_tool_impl(
    guidelines: List[Dict[str, Any]],
    max_chars: int = 1200,
) -> Dict[str, Any]:
    """
    Simple concatenation-based summarizer for insurance guideline chunks.
    """
    if not guidelines:
        return {"summary": "No guideline content found."}

    pieces: List[str] = []
    total_len = 0

    for g in guidelines:
        header = f"{g.get('title', '')} – {g.get('section_title', '')}"
        text = (g.get("text") or "").strip().replace("\n", " ")
        chunk = f"[{header}] {text}"
        if total_len + len(chunk) > max_chars:
            break
        pieces.append(chunk)
        total_len += len(chunk)

    summary = " ".join(pieces)
    return {"summary": summary}


def workflow_step_extractor_tool_impl(text: str) -> Dict[str, Any]:
    """
    Very simple heuristic workflow step extractor.
    Used for things like 'motor claim processing', 'policy lapse reinstatement' etc.
    """
    raw_sentences = re.split(r"[.]\s+", text)
    steps: List[str] = []
    for s in raw_sentences:
        s = s.strip(" .\n\t")
        if not s:
            continue
        steps.append(s)

    formatted = [f"Step {i+1}: {txt}" for i, txt in enumerate(steps)]
    return {"steps": formatted}


def premium_calculator_tool_impl(
    question: str,
    base_premium: Optional[float] = None,
    risk_factor: Optional[float] = None,
    discount_percentage: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Demo-only insurance premium calculator.

    - If base_premium not provided, use a simple heuristic based on question context.
    - If risk_factor is not provided, infer from words like 'high risk', 'low risk'.
    - If discount_percentage not provided, infer if 'no-claim bonus' or 'discount' mentioned.

    NOTE: This is NOT a real pricing engine; it's for demo purposes only.
    """
    q = question.lower()

    # Infer base premium heuristic
    if base_premium is None:
        if "motor" in q or "vehicle" in q:
            base_premium = 10000.0
        elif "health" in q:
            base_premium = 15000.0
        elif "life" in q:
            base_premium = 20000.0
        else:
            base_premium = 12000.0

    # Infer risk factor
    if risk_factor is None:
        if "high risk" in q:
            risk_factor = 1.5
        elif "low risk" in q:
            risk_factor = 0.8
        else:
            risk_factor = 1.0

    # Infer discount
    if discount_percentage is None:
        if "no-claim bonus" in q or "no claim bonus" in q:
            discount_percentage = 10.0
        elif "discount" in q:
            discount_percentage = 5.0
        else:
            discount_percentage = 0.0

    gross_premium = base_premium * risk_factor
    discount_amount = gross_premium * (discount_percentage / 100.0)
    final_premium = gross_premium - discount_amount

    return {
        "base_premium": round(base_premium, 2),
        "risk_factor": round(risk_factor, 2),
        "gross_premium": round(gross_premium, 2),
        "discount_percentage": round(discount_percentage, 2),
        "discount_amount": round(discount_amount, 2),
        "final_premium": round(final_premium, 2),
        "note": "Demo-only premium calculation; for illustration purposes, not actual pricing.",
    }


def safety_reflection_tool_impl(
    question: str,
    guideline_summary: str,
    draft_answer: str,
) -> Dict[str, Any]:
    """
    LLM-based reflection tool for INSURANCE:
    - Ensures draft answer is supported by guideline_summary.
    - Fixes missing or unclear content.
    - Ensures no legal/financial advice beyond documented policy rules.
    """
    system_msg = (
        "You are a strict reviewer of an internal insurance policy assistant.\n"
        "- Ensure the assistant's draft answer is fully supported by the guideline summary.\n"
        "- Do NOT introduce new policy rules or promises that are not implied by the summary.\n"
        "- If important information is missing or unclear, fix it based on the summary.\n"
        "- Use a clear, professional tone suitable for internal operations teams.\n"
        "- Always include the disclaimer: "
        "'This summary is based on internal policy guidelines and does not constitute legal or financial advice.'"
    )

    user_msg = (
        f"User question:\n{question}\n\n"
        f"Guideline summary:\n{guideline_summary}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        "Task: Return the best possible final answer that is safe, precise, and fully supported "
        "by the guideline summary."
    )

    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.1,
    )

    final_answer = resp.choices[0].message.content or ""

    return {
        "final_answer": final_answer,
    }


def audit_log_tool_impl(
    question: str,
    tools_used: List[str],
    final_answer_preview: str,
) -> Dict[str, Any]:
    """
    Demo-only audit logger: prints to console.
    """
    print("\n[AUDIT LOG] ---")
    print(f"Question   : {question}")
    print(f"Tools used : {tools_used}")
    print(f"Answer preview (first 200 chars): {final_answer_preview[:200]!r}")
    print("[AUDIT LOG] ---\n")

    return {
        "logged": True,
    }


# ----------------- TOOL SCHEMA (OpenAI tools=...) -----------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "route_classifier_tool",
            "description": (
                "Classify the user question into an insurance department and intent type "
                "(guideline / workflow / premium_calculation / document_requirements)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The raw user question.",
                    }
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_guidelines_tool",
            "description": (
                "Search internal insurance guidelines in the vector database. "
                "Use this to retrieve relevant sections for answering the question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search query for guidelines.",
                    },
                    "department": {
                        "type": "string",
                        "description": (
                            "Optional department name like 'Underwriting', 'Claims', "
                            "'Fraud Control', 'Customer Service', 'Life Claims', 'General Insurance'."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of guideline sections to retrieve.",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_chunks_tool",
            "description": (
                "Summarize a list of insurance guideline chunks into a shorter text for reasoning."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "guidelines": {
                        "type": "array",
                        "description": "List of guideline result objects as returned by search_guidelines_tool.",
                        "items": {"type": "object"},
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum character length of the summary.",
                        "default": 1200,
                    },
                },
                "required": ["guidelines"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "workflow_step_extractor_tool",
            "description": (
                "Extracts step-by-step workflow steps from a guideline summary or long text "
                "for processes like claims, renewals, or servicing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The guideline summary or relevant text describing the workflow.",
                    }
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "premium_calculator_tool",
            "description": (
                "Demo-only insurance premium calculator that applies simple heuristic "
                "base premium, risk factor, and discount to illustrate pricing logic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original user question (for inferring risk / discount from text).",
                    },
                    "base_premium": {
                        "type": "number",
                        "description": "Base premium before adjustments. If omitted, inferred from question.",
                    },
                    "risk_factor": {
                        "type": "number",
                        "description": "Risk multiplier (e.g., 1.2 for moderate risk). If omitted, inferred.",
                    },
                    "discount_percentage": {
                        "type": "number",
                        "description": "Discount percentage, e.g. for no-claim bonus.",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "safety_reflection_tool",
            "description": (
                "Safety and reflection tool. Checks whether the draft answer is fully supported "
                "by the guideline summary, and returns a refined final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original user question.",
                    },
                    "guideline_summary": {
                        "type": "string",
                        "description": "Summary of relevant guideline content.",
                    },
                    "draft_answer": {
                        "type": "string",
                        "description": "Draft answer to be reviewed and improved.",
                    },
                },
                "required": ["question", "guideline_summary", "draft_answer"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "audit_log_tool",
            "description": (
                "Audit logging tool. Records which tools were used and a short preview of the answer. "
                "Demo-only: prints to console."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The original user question.",
                    },
                    "tools_used": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Names of tools used in this conversation.",
                    },
                    "final_answer_preview": {
                        "type": "string",
                        "description": "First few hundred characters of the answer.",
                    },
                },
                "required": ["question", "tools_used", "final_answer_preview"],
            },
        },
    },
]


# ----------------- AGENT LOOP (no LangGraph) -----------------

SYSTEM_PROMPT = (
    "You are an internal insurance policy assistant for underwriting, claims, fraud control, "
    "customer service, and life insurance teams.\n"
    "- You have access to several tools:\n"
    "  1) route_classifier_tool: decide department and intent.\n"
    "  2) search_guidelines_tool: retrieve internal insurance guideline sections.\n"
    "  3) summarize_chunks_tool: summarize guideline sections.\n"
    "  4) workflow_step_extractor_tool: produce clear step-by-step workflows.\n"
    "  5) premium_calculator_tool: demo-only premium calculator.\n"
    "  6) safety_reflection_tool: review and refine a draft answer.\n"
    "  7) audit_log_tool: log the interaction (demo-only).\n"
    "- Typical sequence for a policy or claims question:\n"
    "  a) Call route_classifier_tool.\n"
    "  b) Call search_guidelines_tool with an appropriate query and department.\n"
    "  c) Call summarize_chunks_tool on the retrieved results.\n"
    "  d) For workflow questions, call workflow_step_extractor_tool on the summary.\n"
    "  e) For premium/pricing questions, call premium_calculator_tool.\n"
    "  f) Form a DRAFT answer using the guideline summary (and steps/premium details if used).\n"
    "  g) Call safety_reflection_tool with the question, summary, and DRAFT answer.\n"
    "  h) Use the final_answer from safety_reflection_tool as the final user-facing answer.\n"
    "  i) Optionally call audit_log_tool at the end.\n"
    "- DO NOT answer the user directly until you have called safety_reflection_tool.\n"
    "- Your final answer (with no further tool calls) must be the refined answer after reflection.\n"
    "- Always include this disclaimer at the end:\n"
    "  'This summary is based on internal policy guidelines and does not constitute legal or financial advice.'"
)


def run_agent(question: str) -> str:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tools_used: List[str] = []
    last_guidelines: List[Dict[str, Any]] = []
    last_summary: str = ""
    final_answer: str = ""

    while True:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        msg = response.choices[0].message

        # Log assistant content (if any)
        if msg.content:
            print("\n[ASSISTANT] (thinking) ->")
            print(msg.content[:300], "...\n")

        tool_calls = msg.tool_calls

        if tool_calls:
            # Assistant requested tools
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": tool_calls,
                }
            )

            for tool_call in tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}

                tools_used.append(tool_name)

                print(f"[AGENT] Tool called: {tool_name} with args={args}")

                # Dispatch to the correct implementation
                if tool_name == "route_classifier_tool":
                    result = route_classifier_tool_impl(**args)

                elif tool_name == "search_guidelines_tool":
                    result = search_guidelines_tool_impl(**args)
                    last_guidelines = result.get("results", [])

                elif tool_name == "summarize_chunks_tool":
                    # guidelines is expected as a list of objects
                    if "guidelines" not in args or not args["guidelines"]:
                        args["guidelines"] = last_guidelines
                    result = summarize_chunks_tool_impl(**args)
                    last_summary = result.get("summary", "")

                elif tool_name == "workflow_step_extractor_tool":
                    result = workflow_step_extractor_tool_impl(**args)

                elif tool_name == "premium_calculator_tool":
                    result = premium_calculator_tool_impl(**args)

                elif tool_name == "safety_reflection_tool":
                    # we also track final answer from reflection
                    result = safety_reflection_tool_impl(**args)
                    final_answer = result.get("final_answer", "")

                elif tool_name == "audit_log_tool":
                    result = audit_log_tool_impl(**args)

                else:
                    result = {"error": f"Unknown tool: {tool_name}"}

                print(f"[AGENT] Tool result (preview): {str(result)[:2000]}...\n")

                # Add tool result message back to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": json.dumps(result),
                    }
                )

            # Loop to let the model see the tool results and decide next step
            continue

        else:
            # No tool calls: this should be the final user-facing answer
            final_answer = msg.content or ""
            break

    return final_answer


def main():
    print("Insurance Agentic Assistant (Azure OpenAI tools + Qdrant)")
    print("Type your question, or 'exit' to quit.\n")

    print("Example questions you can try:")
    print("  1) What are the eligibility criteria for a standard health insurance applicant?")
    print("  2) What are the steps in processing a motor insurance claim from registration to settlement?")
    print("  3) What is the process to reinstate a lapsed health insurance policy after the grace period?")
    print("  4) What documents are required to settle an accidental death claim under a life insurance policy?\n")

    while True:
        question = input("Question> ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        answer = run_agent(question)
        print("\n=== FINAL ANSWER ===")
        print(answer)
        print("====================\n")


if __name__ == "__main__":
    main()