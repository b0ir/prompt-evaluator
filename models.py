from pydantic import BaseModel, Field


class EvaluateRequest(BaseModel):
    prompt: str = Field(
        description=(
            "The actual prompt text you want to evaluate — what you'd paste as a system prompt. "
            "This is the thing being scored and improved."
        )
    )
    task_description: str | None = Field(
        default=None,
        description=(
            "Short plain-English description of what the prompt is supposed to do "
            "(e.g. 'Summarize customer emails into bullet points'). "
            "Used to generate test cases and grade outputs. "
            "If omitted, it is auto-inferred from the prompt."
        ),
    )
    num_test_cases: int = Field(default=3, ge=1, le=5)
    baseline_prompt: str | None = Field(
        default=None,
        description="Optional previous prompt version. Enables score_delta comparison.",
    )
    output_format: str | None = Field(
        default=None,
        description="Output format for syntax validation: 'python', 'json', or 'regex'. Auto-detected if omitted.",
    )


class DimensionScore(BaseModel):
    score: float
    notes: str


class EvaluateResponse(BaseModel):
    overall_score: float
    structural_score: float
    output_score: float
    dimensions: dict[str, DimensionScore]
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    improved_prompt: str
    baseline_score: float | None = None
    score_delta: float | None = None
    dataset_cached: bool = False
