import ast
import asyncio
import hashlib
import json
import re as re_module
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import litellm
from litellm.exceptions import BadRequestError, UnsupportedParamsError


class AsyncEvaluator:
    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache"
        self._supports_response_format: bool | None = None

    def _base_kwargs(self) -> dict:
        kw: dict = {}
        if self.api_base:
            kw["api_base"] = self.api_base
        if self.api_key:
            kw["api_key"] = self.api_key
        return kw

    # ─── JSON helper ───────────────────────────────────────────────────────────

    async def _json_call(self, messages: list[dict], system: str = "") -> dict:
        full_system = system + "\n\nYou MUST respond with valid JSON only. No prose, no markdown fences."
        for attempt in range(2):
            kwargs: dict = dict(
                model=self.model,
                messages=[{"role": "system", "content": full_system}] + messages,
                temperature=0.0 if attempt == 1 else 0.3,
                max_tokens=2000,
                **self._base_kwargs(),
            )
            use_rf = self._supports_response_format is not False
            if use_rf:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                resp = await litellm.acompletion(**kwargs)
                if self._supports_response_format is None:
                    self._supports_response_format = True
            except (BadRequestError, UnsupportedParamsError) as e:
                if use_rf and ("response_format" in str(e) or "json" in str(e).lower()):
                    self._supports_response_format = False
                    kwargs.pop("response_format", None)
                    resp = await litellm.acompletion(**kwargs)
                else:
                    raise

            raw = resp.choices[0].message.content or ""
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                cleaned = re_module.sub(r"```json?\s*|\s*```", "", raw).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    if attempt == 1:
                        raise ValueError(f"JSON parse failed after 2 attempts: {raw[:300]}")

        raise RuntimeError("unreachable")

    # ─── Code grader ───────────────────────────────────────────────────────────

    def _detect_output_format(self, task_description: str) -> str | None:
        td = task_description.lower()
        if any(w in td for w in ["python", "function", "class", "script", " def ", "code", "method"]):
            return "python"
        if any(w in td for w in ["json object", "json array", "json schema", "valid json", "yaml"]):
            return "json"
        if any(w in td for w in ["regex", "regular expression", "regexp", "pattern match"]):
            return "regex"
        return None

    def _extract_code_block(self, output: str) -> str:
        match = re_module.search(r"```[\w]*\n?(.*?)```", output, re_module.DOTALL)
        return match.group(1).strip() if match else output.strip()

    def _code_grade(self, output: str, output_format: str | None) -> float | None:
        """Returns 0-10 syntax score, or None if format unknown/undetected."""
        if not output_format:
            return None
        code = self._extract_code_block(output)
        if output_format == "python":
            try:
                ast.parse(code)
                return 10.0
            except SyntaxError:
                return 0.0
        elif output_format == "json":
            try:
                json.loads(code)
                return 10.0
            except json.JSONDecodeError:
                return 0.0
        elif output_format == "regex":
            try:
                re_module.compile(code)
                return 10.0
            except re_module.error:
                return 0.0
        return None

    # ─── Dataset caching ───────────────────────────────────────────────────────

    def _cache_key(self, task_description: str, num_test_cases: int) -> str:
        content = f"{task_description.strip()}:{num_test_cases}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def _get_test_cases(
        self, task_description: str, num_test_cases: int
    ) -> tuple[list[dict], bool]:
        """Returns (test_cases, was_cached). Generates + persists on cache miss."""
        key = self._cache_key(task_description, num_test_cases)
        path = self.cache_dir / f"{key}.json"

        if path.exists():
            with open(path) as f:
                return json.load(f)["test_cases"], True

        test_cases = await self._generate_test_cases(task_description, num_test_cases)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(
                {
                    "task_description": task_description,
                    "num_test_cases": num_test_cases,
                    "test_cases": test_cases,
                    "created_at": datetime.utcnow().isoformat(),
                },
                f,
                indent=2,
            )
        return test_cases, False

    # ─── Task description inference ────────────────────────────────────────────

    async def _infer_task_description(self, prompt: str) -> str:
        user_msg = dedent(f"""
            Read the following prompt and write ONE sentence describing what it asks an AI to do.
            Be concise and specific. Describe the task, not the format.

            <prompt>
            {prompt}
            </prompt>

            Respond with this exact JSON shape:
            {{"task_description": "<one sentence>"}}
        """).strip()

        result = await self._json_call(
            messages=[{"role": "user", "content": user_msg}],
            system="You extract the core task from a prompt in one sentence.",
        )
        return result.get("task_description", prompt[:200])

    async def _refine_task_description(self, task_description: str, prompt: str) -> str:
        user_msg = dedent(f"""
            A user wants to evaluate the following prompt and provided a task description hint.
            The hint may be too vague to generate rigorous, diverse test cases.

            Prompt being evaluated:
            <prompt>
            {prompt}
            </prompt>

            User's task description hint:
            <hint>
            {task_description}
            </hint>

            Rewrite the task description as ONE specific, concrete sentence that:
            - Preserves the user's original intent from the hint
            - Uses context from the prompt to add specificity
            - Is precise enough to generate diverse, meaningful test cases
            - Names the actual task, input type, and expected output where discernible

            If the hint is already specific and complete, return it unchanged.

            Respond with this exact JSON shape:
            {{"task_description": "<refined one sentence>"}}
        """).strip()

        result = await self._json_call(
            messages=[{"role": "user", "content": user_msg}],
            system="You refine vague task descriptions into specific, testable sentences using prompt context.",
        )
        return result.get("task_description", task_description)

    # ─── LLM pipeline steps ────────────────────────────────────────────────────

    async def _structural_eval(self, prompt: str) -> dict:
        user_msg = dedent(f"""
            You are grading a prompt on 7 dimensions. For each dimension, check the specific criteria
            listed, score 1-10, and write brief notes citing what is present or missing.
            This prompt is intended for Claude Code — an agentic AI software engineer.

            Prompt to evaluate:
            <prompt>
            {prompt}
            </prompt>

            DIMENSION RUBRICS — check each criterion explicitly:

            CLARITY (technique: "Be Clear and Direct")
            Check: Does the first line start with an action verb? (Generate/Write/Create/Identify/Analyze/Summarize/Extract/List)
            Check: Is the task phrased as an instruction/command, NOT as a question?
            Check: Is there zero ambiguity about what the AI should produce?
            Score 9-10: action verb on first line + command phrasing + no ambiguity
            Score 6-8: task is clear but missing action verb, or mild ambiguity
            Score 1-5: phrased as a question, or genuinely unclear what to do

            SPECIFICITY (technique: "Be Specific")
            Check: Are output quality guidelines present? (e.g. "use bullet points", "in under 200 words", "include a docstring")
            Check: For multi-step tasks, are process steps defined? (e.g. "First identify X, then summarize Y, finally output Z")
            Check: Does the prompt define what a GOOD output looks like, not just what to do?
            Score 9-10: has output quality guidelines AND process steps
            Score 6-8: has output quality guidelines but no process steps, or vice versa
            Score 1-5: just a task statement with no quality or process guidance

            CONTEXT (persona and background)
            Check: Is a role or persona explicitly assigned? (e.g. "You are a Python expert", "You are a customer support agent")
            Check: Is relevant background context provided when the task needs it?
            Score 9-10: explicit persona + relevant background context
            Score 6-8: some context but no explicit role assignment
            Score 1-5: no persona, no background — blank slate instruction

            EXAMPLES (technique: "Provide Examples")
            Check: Is at least one concrete example provided?
            Check: Are examples wrapped in XML tags? (e.g. <example>, <sample_input>/<ideal_output>)
            Check: Do examples illustrate the expected output format and quality level?
            Score 9-10: multiple XML-tagged examples covering typical + edge cases
            Score 6-8: one example present but not XML-tagged, or not annotated
            Score 1-5: no examples at all

            STRUCTURE (technique: "Use XML Tags")
            Check: Are XML tags used to separate distinct content? (instructions, data, examples, format spec)
            Check: Is there a logical flow? (role → context → task → requirements → format)
            Check: Is mixed content (instructions + data + examples) clearly delimited?
            Score 9-10: XML tags throughout, clear section hierarchy
            Score 6-8: some separation but mixed content areas without tags
            Score 1-5: unstructured wall of text, no delimiters

            CONSTRAINTS (technique: "Specify Output Quality Guidelines")
            Check: Is the output FORMAT specified? (JSON, bullet list, numbered list, prose, code block)
            Check: Is LENGTH or SCOPE constrained? (max words, number of items, one paragraph only)
            Check: Are TONE and STYLE stated? (formal, concise, technical, beginner-friendly)
            Score 9-10: format + length + tone all specified
            Score 6-8: format specified but length or tone missing
            Score 1-5: no format, no length, no tone constraints

            AGENTIC SCOPE (Claude Code — scope boundary and success condition)
            This prompt will be given to Claude Code, an agentic AI that reads files, edits code, and runs commands.
            Check: Is the scope boundary explicit? Does the prompt state what NOT to change/add/touch?
            Check: Is there a success/terminal condition? Does the prompt define what "done" looks like?
            Check: Does the prompt prevent scope creep? (no unrequested features, no refactoring beyond the task)
            Check: For multi-step tasks — are the deliverables enumerated? Does Claude Code know what files/outputs to produce?
            Score 9-10: explicit do/don't boundary + clear success condition + deliverables listed
            Score 6-8: task is reasonably scoped but missing an explicit boundary or success condition
            Score 1-5: open-ended instruction with no scope constraint — Claude Code could overbuild indefinitely

            Respond with this exact JSON shape:
            {{
              "dimensions": {{
                "clarity":       {{"score": <1-10>, "notes": "<specific finding — cite what is present or missing>"}},
                "specificity":   {{"score": <1-10>, "notes": "<specific finding>"}},
                "context":       {{"score": <1-10>, "notes": "<specific finding>"}},
                "examples":      {{"score": <1-10>, "notes": "<specific finding>"}},
                "structure":     {{"score": <1-10>, "notes": "<specific finding>"}},
                "constraints":   {{"score": <1-10>, "notes": "<specific finding>"}},
                "agentic_scope": {{"score": <1-10>, "notes": "<specific finding>"}}
              }},
              "overall_structural_score": <mean of 7 scores as float>
            }}
        """).strip()

        result = await self._json_call(
            messages=[{"role": "user", "content": user_msg}],
            system=(
                "You are an expert prompt engineer specialising in prompts for Claude Code — an agentic AI software engineer. "
                "Grade prompts rigorously using the exact criteria given. "
                "Be strict — a prompt must explicitly contain what is asked for to score high."
            ),
        )

        # Defensive recompute
        try:
            scores = [v["score"] for v in result["dimensions"].values()]
            result["overall_structural_score"] = sum(scores) / len(scores)
        except (KeyError, TypeError, ZeroDivisionError):
            pass

        return result

    async def _generate_test_cases(self, task_description: str, n: int) -> list[dict]:
        user_msg = dedent(f"""
            Generate {n} unique, diverse test cases for evaluating a prompt given to Claude Code — an agentic AI software engineer that reads files, edits code, runs terminal commands, and must operate within a real codebase.

            <task_description>
            {task_description}
            </task_description>

            For each test case provide:
            - "scenario": a specific coding/engineering context — name the project type, the current state of the code, and what the developer needs. Make it feel like a real developer request, not an abstract test.
            - "solution_criteria": 2-4 specific, checkable requirements the output MUST satisfy. Include at least one criterion about scope discipline (no unrequested changes) when relevant.

            Cover different aspects: a typical straightforward case, an edge case or ambiguous input, and (if n≥3) a case where scope creep is a real risk.

            Respond with this exact JSON shape:
            {{
              "test_cases": [
                {{
                  "scenario": "...",
                  "solution_criteria": ["criterion 1", "criterion 2"]
                }}
              ]
            }}
        """).strip()

        result = await self._json_call(
            messages=[{"role": "user", "content": user_msg}],
            system=(
                "You are a test case designer for Claude Code prompts. "
                "Generate realistic software engineering scenarios that mirror how developers actually use Claude Code. "
                "Be specific about project context, tech stack, and current code state."
            ),
        )

        cases = result.get("test_cases", [])
        if not cases:
            for v in result.values():
                if isinstance(v, list):
                    return v
        return cases

    async def _run_scenario(self, prompt: str, scenario: str) -> str:
        try:
            resp = await litellm.acompletion(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Scenario: {scenario}"},
                ],
                temperature=0.7,
                max_tokens=1500,
                **self._base_kwargs(),
            )
            return resp.choices[0].message.content or "[ERROR: empty output]"
        except Exception as e:
            return f"[ERROR: scenario run failed — {e}]"

    async def _grade_output(
        self, task_description: str, test_case: dict, output: str
    ) -> dict:
        scenario = test_case["scenario"]
        criteria = test_case.get("solution_criteria", [])
        criteria_str = (
            "\n".join(f"- {c}" for c in criteria)
            if criteria
            else "- Output adequately addresses the scenario"
        )

        user_msg = dedent(f"""
            Your task is to evaluate the following AI-generated output with EXTREME RIGOR.

            Original task description:
            <task_description>
            {task_description}
            </task_description>

            Test scenario:
            <scenario>
            {scenario}
            </scenario>

            Output to evaluate:
            <output>
            {output}
            </output>

            Criteria the output MUST satisfy:
            <solution_criteria>
            {criteria_str}
            </solution_criteria>

            Mandatory Requirements - ANY VIOLATION MEANS AUTOMATIC FAILURE (score of 3 or lower):
            - Output must directly address the test scenario
            - Output must satisfy every item in solution_criteria

            Scoring Guidelines:
            * Score 1-3: Fails one or more mandatory requirements
            * Score 4-6: Meets mandatory requirements but significant quality issues
            * Score 7-8: Meets all requirements with minor issues
            * Score 9-10: Meets all requirements excellently

            Respond with this exact JSON shape:
            {{
              "strengths": ["<strength 1>", "<strength 2>"],
              "weaknesses": ["<weakness 1>", "<weakness 2>"],
              "reasoning": "<concise explanation>",
              "score": <integer 1-10>
            }}
        """).strip()

        try:
            return await self._json_call(
                messages=[{"role": "user", "content": user_msg}],
                system="You are an expert evaluator. Grade AI outputs with extreme rigor.",
            )
        except Exception:
            return {
                "strengths": [],
                "weaknesses": ["Evaluation failed"],
                "reasoning": "Error during grading.",
                "score": 1,
            }

    async def _synthesize(
        self, prompt: str, structural_result: dict, grades: list[dict]
    ) -> dict:
        user_msg = dedent(f"""
            You have just evaluated a prompt. Synthesize the findings and rewrite the prompt.

            Original prompt:
            <prompt>
            {prompt}
            </prompt>

            Structural analysis:
            <structural>
            {json.dumps(structural_result, indent=2)}
            </structural>

            Output quality grades across {len(grades)} test scenarios:
            <output_grades>
            {json.dumps(grades, indent=2)}
            </output_grades>

            TASK 1 — Identify strengths (3-5 items, be specific about what the prompt does well).

            TASK 2 — Identify weaknesses (3-5 items, cite the exact dimension and what is missing).

            TASK 3 — Write suggestions. For each low-scoring dimension, use THESE specific fixes:

            LOW CLARITY → "Start with an action verb on the first line: 'Generate / Write / Summarize / Identify / Analyze [task]'. Remove any question phrasing."
            LOW SPECIFICITY → "Add output quality guidelines after the task statement: 'Format as X. Keep Y under Z words. Include W.' For multi-step tasks, add numbered process steps."
            LOW CONTEXT → "Add a role/persona as the very first sentence: 'You are a [role] expert.' Add background context if the task needs domain knowledge."
            LOW EXAMPLES → "Add at least one XML-tagged example: '<example>\\n<input>...</input>\\n<ideal_output>...</ideal_output>\\n</example>'. Annotate why the output is ideal."
            LOW STRUCTURE → "Wrap distinct sections in XML tags: <task>, <requirements>, <format>, <example>, <constraints>. One section per concern."
            LOW CONSTRAINTS → "Specify output format (JSON / bullet list / prose / code block), max length (X words / Y items), and tone (formal / concise / technical)."
            LOW AGENTIC_SCOPE → "Add an explicit scope boundary: state what NOT to change (e.g. 'Do not modify unrelated files. Do not add features beyond what is specified.'). Add a success condition: what does done look like? (e.g. 'Task is complete when the test suite passes and only the files listed have been changed.')"

            CLAUDE CODE-SPECIFIC patterns to suggest when relevant (only include if applicable to the prompt):
            - If the task is complex and multi-step: "Consider specifying whether Claude Code should use /plan before implementing, or delegate parallel subtasks to subagents via the Agent tool."
            - If the task involves project-specific context: "Add a note to check CLAUDE.md for project conventions before starting, or include the relevant conventions inline."
            - If the task could produce many files: "Enumerate the expected output files explicitly so Claude Code doesn't over-generate."

            TASK 4 — Rewrite the improved_prompt applying ALL of the above fixes.
            Follow this structure in order:
            1. Role line: "You are a [role]." (if needed)
            2. Task line: action verb + specific task (first line after role)
            3. <requirements> tag: output quality guidelines and process steps
            4. <constraints> tag: scope boundary (what NOT to do), success condition, output format, length, tone.
               IMPORTANT — for code tasks: specify architectural organisation (which concerns to separate: models/schemas/crud/routers/auth/etc.) and that each file must be in its own code block with a path comment. Do NOT specify exact file counts — let the model decide how many files the implementation requires.
            5. <example> tag: one XML-tagged example (if examples would help)

            HARD RULES for the improved_prompt:
            - Do NOT add functional requirements, features, or scope that were not present in the original prompt. Only improve how existing requirements are expressed — clarity, structure, specificity, and examples.
            - Any code in <example> blocks must be internally consistent with the constraints and version requirements stated in the prompt (e.g. if the prompt requires Pydantic v2, the example must use Pydantic v2 patterns only).

            Respond with this exact JSON shape:
            {{
              "strengths": ["<specific strength 1>", ...],
              "weaknesses": ["<specific weakness citing dimension 1>", ...],
              "suggestions": ["<concrete fix 1>", ...],
              "improved_prompt": "<full rewritten prompt using the structure above>"
            }}
        """).strip()

        return await self._json_call(
            messages=[{"role": "user", "content": user_msg}],
            system=(
                "You are a senior prompt engineering expert specialising in prompts for Claude Code — an agentic AI software engineer. "
                "Apply the four core techniques: Be Clear and Direct, Be Specific, Provide Examples, Use XML Tags. "
                "For Claude Code prompts, also enforce agentic scope discipline: scope boundaries, success conditions, and scope creep prevention. "
                "Follow instructions exactly. Be concrete — name specific changes, not vague advice."
            ),
        )

    # ─── Internal: run + grade a prompt against all test cases ─────────────────

    async def _run_and_grade(
        self,
        prompt: str,
        task_description: str,
        test_cases: list[dict],
        output_format: str | None,
    ) -> tuple[list[str], list[dict]]:
        raw_outputs = await asyncio.gather(
            *[self._run_scenario(prompt, tc["scenario"]) for tc in test_cases],
            return_exceptions=True,
        )
        outputs = [
            o if not isinstance(o, Exception) else f"[ERROR: {o}]"
            for o in raw_outputs
        ]

        raw_grades = await asyncio.gather(
            *[self._grade_output(task_description, tc, o) for tc, o in zip(test_cases, outputs)],
            return_exceptions=True,
        )
        grades = []
        for i, g in enumerate(raw_grades):
            if isinstance(g, Exception):
                grades.append({
                    "strengths": [],
                    "weaknesses": ["Grading failed"],
                    "reasoning": str(g),
                    "score": 1,
                })
            else:
                # Merge code grader score when output format is known
                code_score = self._code_grade(outputs[i], output_format)
                if code_score is not None:
                    g = dict(g)
                    g["score"] = (float(g.get("score", 1)) + code_score) / 2
                    g["code_syntax_score"] = code_score
                grades.append(g)

        return outputs, grades

    # ─── Public entry point ────────────────────────────────────────────────────

    async def evaluate(
        self,
        prompt: str,
        task_description: str | None,
        num_test_cases: int,
        baseline_prompt: str | None = None,
        output_format: str | None = None,
    ) -> dict:
        num_test_cases = min(num_test_cases, 5)

        # Resolve task_description: infer from scratch if missing, refine if provided but potentially vague
        if not task_description:
            task_description = await self._infer_task_description(prompt)
        else:
            task_description = await self._refine_task_description(task_description, prompt)

        fmt = output_format or self._detect_output_format(task_description)

        # Steps 1+2 in parallel; add baseline structural eval if provided
        if baseline_prompt:
            structural_result, baseline_structural, scenario_result = await asyncio.gather(
                self._structural_eval(prompt),
                self._structural_eval(baseline_prompt),
                self._get_test_cases(task_description, num_test_cases),
            )
        else:
            structural_result, scenario_result = await asyncio.gather(
                self._structural_eval(prompt),
                self._get_test_cases(task_description, num_test_cases),
            )
            baseline_structural = None

        test_cases, was_cached = scenario_result

        # Run + grade main prompt (and baseline in parallel if provided)
        if baseline_prompt:
            (outputs, grades), (_, baseline_grades) = await asyncio.gather(
                self._run_and_grade(prompt, task_description, test_cases, fmt),
                self._run_and_grade(baseline_prompt, task_description, test_cases, fmt),
            )
        else:
            outputs, grades = await self._run_and_grade(prompt, task_description, test_cases, fmt)
            baseline_grades = None

        # Synthesize
        synthesis = await self._synthesize(prompt, structural_result, grades)

        # Compute scores
        structural_score = float(structural_result.get("overall_structural_score", 0))
        output_avg = (
            sum(float(g.get("score", 1)) for g in grades) / len(grades) if grades else 0.0
        )
        overall = round(0.4 * structural_score + 0.6 * output_avg, 2)

        baseline_score = None
        score_delta = None
        if baseline_structural and baseline_grades:
            b_structural = float(baseline_structural.get("overall_structural_score", 0))
            b_output_avg = (
                sum(float(g.get("score", 1)) for g in baseline_grades) / len(baseline_grades)
            )
            baseline_score = round(0.4 * b_structural + 0.6 * b_output_avg, 2)
            score_delta = round(overall - baseline_score, 2)

        test_results = [
            {
                "test_case": tc,
                "output": out,
                "score": float(g.get("score", 1)),
                "reasoning": g.get("reasoning", ""),
                "code_syntax_score": g.get("code_syntax_score"),
            }
            for tc, out, g in zip(test_cases, outputs, grades)
        ]

        return {
            "overall_score": overall,
            "structural_score": round(structural_score, 2),
            "output_score": round(output_avg, 2),
            "dimensions": structural_result.get("dimensions", {}),
            "strengths": synthesis.get("strengths", []),
            "weaknesses": synthesis.get("weaknesses", []),
            "suggestions": synthesis.get("suggestions", []),
            "improved_prompt": synthesis.get("improved_prompt", ""),
            "baseline_score": baseline_score,
            "score_delta": score_delta,
            "dataset_cached": was_cached,
            "_task_description": task_description,   # resolved value (inferred or provided)
            "_test_results": test_results,           # used by /evaluate/report; both stripped by Pydantic on /evaluate
        }
