import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from app.evaluator import AsyncEvaluator
from app.models import EvaluateRequest, EvaluateResponse
from app.report import generate_report

load_dotenv()

_LOCAL_PROVIDERS = {"ollama", "llamacpp", "lm_studio", "huggingface"}

_PROVIDER_KEY_VARS: dict[str, str] = {
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "nvidia_nim": "NVIDIA_NIM_API_KEY",
}


def _resolve_config() -> tuple[str, str | None, str | None]:
    """Returns (model, api_base, api_key). Raises if no key found for cloud providers."""
    model = os.getenv("LLM_MODEL", "openrouter/nvidia/nemotron-3-super-120b-a12b:free")
    api_base = os.getenv("LLM_BASE_URL") or None
    api_key = os.getenv("LLM_API_KEY") or None

    if not api_key:
        provider = model.split("/")[0].lower()
        if provider not in _LOCAL_PROVIDERS:
            env_var = _PROVIDER_KEY_VARS.get(provider, f"{provider.upper()}_API_KEY")
            provider_key = os.getenv(env_var)
            if not provider_key:
                raise RuntimeError(
                    f"No API key for provider '{provider}'. Set {env_var} or LLM_API_KEY in .env"
                )

    return model, api_base, api_key


@asynccontextmanager
async def lifespan(app: FastAPI):
    model, api_base, api_key = _resolve_config()
    app.state.evaluator = AsyncEvaluator(model=model, api_base=api_base, api_key=api_key)
    yield


app = FastAPI(
    title="Prompt Evaluator",
    description="Evaluates any prompt and returns a score, breakdown, suggestions, and improved version.",
    version="0.0.1",
    lifespan=lifespan,
)

_STATIC = Path(__file__).parent.parent / "static"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui():
    return (_STATIC / "index.html").read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest):
    try:
        result = await app.state.evaluator.evaluate(
            prompt=body.prompt,
            task_description=body.task_description,
            num_test_cases=body.num_test_cases,
            baseline_prompt=body.baseline_prompt,
            output_format=body.output_format,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/evaluate/report", response_class=HTMLResponse)
async def evaluate_report(body: EvaluateRequest):
    try:
        result = await app.state.evaluator.evaluate(
            prompt=body.prompt,
            task_description=body.task_description,
            num_test_cases=body.num_test_cases,
            baseline_prompt=body.baseline_prompt,
            output_format=body.output_format,
        )
        return generate_report(
            result, body.prompt, result.get("_task_description", body.task_description or "")
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
