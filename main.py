from __future__ import annotations

import os
import logging
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from openai import AsyncOpenAI, APIError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guardrail")

# --- Configuration (override via environment) -------------------------------
# NOTE: Z.ai's OpenAI-compatible base URL. `https://z.ai` is the marketing
# site, not the API host. Swap MODEL_NAME to "glm-5.2" once it is released.
ZAI_BASE_URL = os.getenv("ZAI_BASE_URL", "https://api.z.ai/api/paas/v4/")
MODEL_NAME = os.getenv("ZAI_MODEL", "glm-4.6")
MAX_TURNS = int(os.getenv("GUARDRAIL_MAX_TURNS", "3"))
GENERATOR_TEMPERATURE = float(os.getenv("GENERATOR_TEMPERATURE", "0.7"))
EVALUATOR_TEMPERATURE = float(os.getenv("EVALUATOR_TEMPERATURE", "0.0"))
REQUEST_TIMEOUT = float(os.getenv("ZAI_TIMEOUT", "60"))

APPROVAL_TOKEN = "APPROVED"

app = FastAPI(
    title="B2B Guardrail Middleware",
    description="Self-correcting generation loop with adversarial evaluation.",
    version="1.0.0",
)


# --- Schemas -----------------------------------------------------------------
class GuardrailRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="End-user prompt.")
    user_zai_key: str = Field(..., min_length=1, description="Caller's Z.ai API key.")


class GuardrailResponse(BaseModel):
    status: str
    turns_used: int
    response: Optional[str] = None
    last_feedback: Optional[str] = None


# --- Prompts -----------------------------------------------------------------
GENERATOR_SYSTEM = (
    "You are a careful B2B assistant. Produce a complete, safe, policy-compliant "
    "answer to the user's request. If you receive evaluator feedback, revise your "
    "previous answer to address every point raised."
)

EVALUATOR_SYSTEM = (
    "You are a strict adversarial reviewer enforcing B2B guardrails: safety, "
    "factual grounding, policy compliance, and no leakage of secrets or PII.\n"
    "Review the candidate answer against the original prompt.\n"
    f"If it fully passes, reply with EXACTLY the single word: {APPROVAL_TOKEN}\n"
    "Otherwise, reply with concise, actionable feedback describing what must change. "
    f"Never reply {APPROVAL_TOKEN} if any issue remains."
)


# --- Z.ai helpers ------------------------------------------------------------
def _client(api_key: str) -> AsyncOpenAI:
    """Build a per-request, OpenAI-compatible Z.ai client.

    The key is caller-supplied, so the client is scoped to the request and
    never cached globally.
    """
    return AsyncOpenAI(
        api_key=api_key,
        base_url=ZAI_BASE_URL,
        timeout=REQUEST_TIMEOUT,
    )


async def _chat(client: AsyncOpenAI, system: str, user: str, temperature: float) -> str:
    resp = await client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


async def _generate(client: AsyncOpenAI, prompt: str, feedback: Optional[str]) -> str:
    if feedback:
        user = (
            f"ORIGINAL REQUEST:\n{prompt}\n\n"
            f"EVALUATOR FEEDBACK ON YOUR PREVIOUS ANSWER:\n{feedback}\n\n"
            "Provide a fully revised answer."
        )
    else:
        user = prompt
    return await _chat(client, GENERATOR_SYSTEM, user, GENERATOR_TEMPERATURE)


async def _evaluate(client: AsyncOpenAI, prompt: str, candidate: str) -> str:
    user = (
        f"ORIGINAL REQUEST:\n{prompt}\n\n"
        f"CANDIDATE ANSWER:\n{candidate}\n\n"
        f"Reply '{APPROVAL_TOKEN}' or give feedback."
    )
    return await _chat(client, EVALUATOR_SYSTEM, user, EVALUATOR_TEMPERATURE)


# --- Payment authorization gate ---------------------------------------------
def _verify_payment_authorization(transaction_id: Optional[str]) -> None:
    """Verify the agent-to-agent payment authorization header.

    In production this would settle/verify the transaction against the ATXP
    network. Here we enforce presence and a minimal format contract.
    """
    if not transaction_id or not transaction_id.strip():
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Missing X-ATXP-Transaction-ID: agent-to-agent payment authorization required.",
        )
    if len(transaction_id.strip()) < 8:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail="Invalid X-ATXP-Transaction-ID format.",
        )


# --- Routes ------------------------------------------------------------------
@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "model": MODEL_NAME}


@app.post("/v1/guardrail", response_model=GuardrailResponse)
async def guardrail(
    body: GuardrailRequest,
    x_atxp_transaction_id: Optional[str] = Header(default=None),
) -> GuardrailResponse:
    _verify_payment_authorization(x_atxp_transaction_id)

    client = _client(body.user_zai_key)
    feedback: Optional[str] = None
    candidate: str = ""

    try:
        for turn in range(1, MAX_TURNS + 1):
            logger.info("Turn %d/%d", turn, MAX_TURNS)

            # Step 1: Generator
            candidate = await _generate(client, body.prompt, feedback)

            # Step 2: Adversarial Evaluator (low temperature)
            verdict = await _evaluate(client, body.prompt, candidate)

            if verdict.strip().upper().startswith(APPROVAL_TOKEN):
                return GuardrailResponse(
                    status="approved",
                    turns_used=turn,
                    response=candidate,
                )

            feedback = verdict
    except APIError as exc:
        logger.exception("Z.ai API error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Upstream model error: {exc}",
        ) from exc
    except httpx.HTTPError as exc:
        logger.exception("Transport error")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Upstream transport error: {exc}",
        ) from exc
    finally:
        await client.close()

    # Step 5: Circuit breaker tripped
    return GuardrailResponse(
        status="rejected_circuit_breaker",
        turns_used=MAX_TURNS,
        response=None,
        last_feedback=feedback,
    )
