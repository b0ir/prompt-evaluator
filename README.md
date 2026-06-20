# Prompt Evaluator

[![License: ELv2](https://img.shields.io/badge/License-Elastic_v2-blue.svg)](LICENSE)

FastAPI service that evaluates any prompt and returns a score, dimension breakdown, improvement suggestions, and a rewritten improved version. Model-agnostic via LiteLLM — works with any provider.

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure your model and API key**

Copy `.env.example` to `.env` and edit:
```bash
cp .env.example .env
```

Set `LLM_MODEL` to any [LiteLLM model string](https://docs.litellm.ai/docs/providers) and the matching API key. Example using OpenRouter (the default):
```
LLM_MODEL=openrouter/owl-alpha
OPENROUTER_API_KEY=sk-or-...
```

To switch providers, set `LLM_MODEL` to any [LiteLLM model string](https://docs.litellm.ai/docs/providers) and the corresponding key:
```
LLM_MODEL=anthropic/claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

For local models (Ollama, etc.) — no key needed:
```
LLM_MODEL=ollama/llama3
LLM_BASE_URL=http://localhost:11434
```

**3. Start the server**
```bash
uvicorn main:app --reload
```

Server runs at `http://localhost:8000`.

---

## Usage

### Browser UI (recommended)
Open `http://localhost:8000/` — paste your prompt, adjust options, hit **Evaluate** or **Open HTML Report**. No JSON syntax required.

### Interactive API docs
Open `http://localhost:8000/docs` for Swagger UI.

---

### `POST /evaluate`

Evaluates a prompt using both structural analysis and output-based testing.

**`prompt` vs `task_description` — what's the difference?**

- **`prompt`** — the actual text being evaluated. What you'd paste as a system prompt.
- **`task_description`** — used **internally** to generate test cases and grade outputs. The evaluator needs to know what the prompt is supposed to do so it can create relevant scenarios and judge responses. It is **never passed to the model being evaluated** — it's context for the evaluator only. If omitted, it is auto-inferred from the prompt.

**How to write a good task_description**

Write it as a direct task sentence, not as a description of the prompt:

| ❌ Weak | ✅ Strong |
|---|---|
| `"make a plan"` | `"Generate a step-by-step project plan with milestones and owners"` |
| `"help with FastAPI"` | `"Answer FastAPI questions with code examples and explain each step"` |
| `"this prompt should do CRUD"` | `"Generate a FastAPI backend with CRUD endpoints for a User resource, including Pydantic schemas, SQLAlchemy models, and 404/422 error handling"` |

Rules:
- Start with an action verb (`Generate`, `Write`, `Summarize`, `Identify`, `Answer`)
- Drop "this prompt should" — just describe the task directly
- Name the output type and key requirements (format, scope, constraints)
- The more specific, the better the test cases — and the more meaningful the score

If you provide a vague description, the evaluator will automatically refine it using the full prompt as context before generating test cases.

For simple prompts, `task_description` and `prompt` look similar. For complex ones they diverge:
- `prompt` = `"You are a Python expert. Write a typed function that sorts... <requirements>...</requirements>"`
- `task_description` = `"Generate a typed Python function that sorts a list of integers in ascending order"` ← short goal sentence

**Request body**

| Field | Type | Required | Description |
|---|---|---|---|
| `prompt` | string | ✅ | The actual prompt text to evaluate |
| `task_description` | string | ❌ | Short goal sentence. Auto-inferred from `prompt` if omitted. |
| `num_test_cases` | int | ❌ | Number of test scenarios (1–5, default 3) |
| `baseline_prompt` | string | ❌ | Previous prompt version — enables `score_delta` comparison |
| `output_format` | string | ❌ | `"python"`, `"json"`, or `"regex"` — enables syntax validation. Auto-detected if omitted. |

**Example request — minimal (task_description inferred)**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that sorts a list",
    "num_test_cases": 3
  }'
```

**Example request — explicit task_description**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that sorts a list",
    "task_description": "Generate Python code that sorts a list of integers",
    "num_test_cases": 3
  }'
```

**Example request — with baseline comparison**
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "You are a Python 3.11 expert. Write a clean, typed function that sorts a list of integers in ascending order. Include a docstring and return a new list.",
    "task_description": "Generate Python code that sorts a list of integers",
    "baseline_prompt": "Write a Python function that sorts a list",
    "num_test_cases": 3
  }'
```

**Example response**
```json
{
  "overall_score": 7.8,
  "structural_score": 8.0,
  "output_score": 7.6,
  "dimensions": {
    "clarity":      { "score": 9, "notes": "Clear action verb, specific language and version stated." },
    "specificity":  { "score": 8, "notes": "Output format and return behavior defined." },
    "context":      { "score": 9, "notes": "Persona and expertise level set explicitly." },
    "examples":     { "score": 3, "notes": "No examples provided — edge cases left to model." },
    "structure":    { "score": 8, "notes": "Single block, could use XML tags for requirements." },
    "constraints":  { "score": 7, "notes": "Return behavior specified; algorithm type not constrained." }
  },
  "strengths": ["Clear persona", "Explicit return behavior", "Language version specified"],
  "weaknesses": ["No examples for edge cases", "Missing XML structure for requirements"],
  "suggestions": [
    "Add a worked example with sample input and output",
    "Wrap requirements in <requirements> XML tags",
    "Specify algorithm constraints if relevant (e.g. no built-ins)"
  ],
  "improved_prompt": "You are a Python 3.11 expert. Write a clean, well-documented function...",
  "baseline_score": 4.2,
  "score_delta": 3.6,
  "dataset_cached": false
}
```

`score_delta` is positive when your new prompt outperforms the baseline. The **same test cases** are used for both — scores are directly comparable.

---

### How scoring works

Each request runs **9 LLM calls** (for `num_test_cases=3`):

| Step | Calls | What happens |
|---|---|---|
| Structural eval | 1 | Scores prompt on 6 dimensions (clarity, specificity, context, examples, structure, constraints) |
| Generate test cases | 1 | Creates N structured test cases with explicit `solution_criteria` per case |
| Run scenarios | N | Runs the prompt against each scenario |
| Grade outputs | N | Grades each output against its `solution_criteria` — any violation → score ≤ 3 |
| Synthesize | 1 | Produces strengths, weaknesses, suggestions, improved prompt |

With `baseline_prompt`: adds 1 extra structural eval + N extra scenario runs + N extra grades (all parallelized).

```
overall_score = 0.4 × structural_score + 0.6 × mean(output_grades)
```

When `output_format` is set (or auto-detected), output grades are combined with a **syntax score**:
```
output_grade = (model_score + syntax_score) / 2
```
Syntax score is 10 if the output parses cleanly (valid Python/JSON/regex), 0 if it fails.

### Dataset caching

Test cases for a given `task_description` + `num_test_cases` are saved to `cache/` after the first request. Subsequent requests reuse them — ensuring fair comparison across prompt iterations and skipping the generation step entirely.

`"dataset_cached": true` in the response means scenarios came from cache.

To regenerate test cases, delete the relevant file from `cache/`.

---

### `POST /evaluate/report`

Same request body as `/evaluate`. Returns a rendered **HTML page** instead of JSON — open directly in a browser or save to file.

```bash
# Save and open
curl -s -X POST http://localhost:8000/evaluate/report \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that sorts a list",
    "task_description": "Generate Python code that sorts a list of integers"
  }' > report.html && open report.html
```

The report includes:
- Summary header: overall / structural / output scores, baseline delta badge, pass rate, test case count
- Prompt evaluated (monospace block)
- Structural dimensions table (6 rows, color-coded scores)
- Per-test-case table: scenario | solution criteria | output | score | reasoning
- Synthesis: strengths / weaknesses / suggestions (3-column layout)
- Improved prompt (monospace block, ready to copy)

Scores are color-coded: green (≥ 8), yellow (6–7), red (≤ 5).

---

### `GET /health`
```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Tips for better evaluations

- Write a specific `task_description` — it drives the quality of test cases and `solution_criteria`. Vague descriptions produce weak tests.
- Use `num_test_cases: 1` for fast feedback during iteration, `3–5` for a thorough eval.
- Always compare against a `baseline_prompt` — absolute scores alone are not meaningful, only the delta is.
- The `improved_prompt` is ready to copy-paste. Submit it as your next iteration with the original as `baseline_prompt`.
- If the prompt generates code/JSON/regex, set `output_format` explicitly or let it auto-detect — syntax validation catches broken outputs the model grader would miss.

---

## License

[Elastic License 2.0](LICENSE) — free to use and modify; cannot be offered as a managed/hosted service to third parties.
