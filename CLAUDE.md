# prompt-evaluator

FastAPI service — structural analysis + output testing + prompt rewriting. Model-agnostic via LiteLLM; defaults to `openrouter/owl-alpha`.

## Dev commands

```bash
# Start server (browser UI at http://localhost:8000/)
uvicorn app.main:app --reload

# Kill port conflict
lsof -ti:8000 | xargs kill -9

# Health check
curl http://localhost:8000/health
```

## Required env

```
LLM_MODEL=openrouter/owl-alpha     # default; any LiteLLM model string works
OPENROUTER_API_KEY=sk-or-...       # set the key for whichever provider LLM_MODEL uses
```

See `.env.example` for all supported providers.

## Architecture

```
app/
  main.py       — FastAPI app + routes (GET /, POST /evaluate, POST /evaluate/report, GET /health)
  evaluator.py  — AsyncEvaluator: 5-step async pipeline
  models.py     — Pydantic request/response models
  report.py     — HTML report renderer
static/         — browser UI (served at GET /)
cache/          — test case cache (sha256-keyed JSON files)
```

## Conventions

- Branch from `main`; open a PR to `main` (don't commit directly). Rebase on `main` before pushing if it moved.
- Never commit `.env`, API keys, `cache/`, or logs — all gitignored. Keep it that way.

## Non-obvious behaviors

- **Dataset cache**: keyed by `sha256(task_description + num_test_cases)[:16]` → `cache/<hash>.json`. Delete file to regenerate.
- **task_description refinement**: when user provides a task_description, `_refine_task_description()` enriches it via LLM before test case generation — invalidates old cache entries for that description.
- **Scoring**: `overall = 0.4 × structural_score + 0.6 × mean(output_grades)`. Total LLM calls: `2N+3` (N=3 → 9).
- **Syntax scoring**: when `output_format` set or auto-detected, grade becomes `(model_score + syntax_score) / 2` where syntax_score is 10 (pass) or 0 (fail).
- **Synthesize HARD RULES**: no scope creep (no features absent from original prompt), example code must match version constraints stated in the prompt.
