from typing import Any


def generate_report(eval_result: dict[str, Any], prompt: str, task_description: str) -> str:
    test_results = eval_result.get("_test_results", [])
    overall = eval_result.get("overall_score", 0)
    structural = eval_result.get("structural_score", 0)
    output_score = eval_result.get("output_score", 0)
    score_delta = eval_result.get("score_delta")
    dimensions = eval_result.get("dimensions", {})
    strengths = eval_result.get("strengths", [])
    weaknesses = eval_result.get("weaknesses", [])
    suggestions = eval_result.get("suggestions", [])
    improved_prompt = eval_result.get("improved_prompt", "")
    dataset_cached = eval_result.get("dataset_cached", False)

    total_tests = len(test_results)
    pass_rate = (
        100 * len([r for r in test_results if r["score"] >= 7]) / total_tests if total_tests else 0
    )

    def score_class(s: float) -> str:
        if s >= 8:
            return "score-high"
        if s <= 5:
            return "score-low"
        return "score-medium"

    def score_badge(s: float) -> str:
        return f'<span class="score {score_class(s)}">{s:.1f}</span>'

    # Baseline delta badge
    delta_html = ""
    if score_delta is not None:
        sign = "+" if score_delta >= 0 else ""
        color = "#2e7d32" if score_delta >= 0 else "#c62828"
        bg = "#c8e6c9" if score_delta >= 0 else "#ffcdd2"
        delta_html = f'<span class="score" style="background:{bg};color:{color}">{sign}{score_delta:.2f} vs baseline</span>'

    # Per-test-case rows
    test_rows = ""
    for r in test_results:
        tc = r.get("test_case", {})
        criteria_html = "<br>• ".join(tc.get("solution_criteria", []))
        output_escaped = r.get("output", "").replace("<", "&lt;").replace(">", "&gt;")
        code_syntax = r.get("code_syntax_score")
        syntax_note = (
            f'<br><small style="color:#666">syntax: {code_syntax:.0f}/10</small>'
            if code_syntax is not None
            else ""
        )
        test_rows += f"""
        <tr>
            <td>{tc.get("scenario", "")}</td>
            <td class="criteria">{"• " + criteria_html if criteria_html else "—"}</td>
            <td class="output"><pre>{output_escaped}</pre></td>
            <td class="score-col">{score_badge(r.get("score", 0))}{syntax_note}</td>
            <td>{r.get("reasoning", "")}</td>
        </tr>"""

    # Dimension rows
    dim_rows = ""
    for name, data in dimensions.items():
        s = data.get("score", 0) if isinstance(data, dict) else getattr(data, "score", 0)
        notes = data.get("notes", "") if isinstance(data, dict) else getattr(data, "notes", "")
        dim_rows += f"""
        <tr>
            <td style="font-weight:bold;text-transform:capitalize">{name}</td>
            <td>{score_badge(s)}</td>
            <td>{notes}</td>
        </tr>"""

    strengths_html = "".join(f"<li>{s}</li>" for s in strengths)
    weaknesses_html = "".join(f"<li>{w}</li>" for w in weaknesses)
    suggestions_html = "".join(f"<li>{s}</li>" for s in suggestions)
    improved_escaped = improved_prompt.replace("<", "&lt;").replace(">", "&gt;")
    prompt_escaped = prompt.replace("<", "&lt;").replace(">", "&gt;")
    cached_badge = (
        '<span style="font-size:12px;color:#666;margin-left:8px">(dataset cached)</span>'
        if dataset_cached
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Prompt Evaluation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }}
    h2 {{ margin-top: 32px; border-bottom: 2px solid #ddd; padding-bottom: 6px; }}
    .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
    .summary-stats {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px; }}
    .stat-box {{ background: #fff; border-radius: 5px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,.1); min-width: 160px; }}
    .stat-value {{ font-size: 24px; font-weight: bold; margin-top: 4px; }}
    .prompt-box {{ background: #f8f8f8; border-left: 4px solid #4a4a4a; padding: 12px 16px; font-family: monospace; white-space: pre-wrap; font-size: 13px; border-radius: 3px; margin: 8px 0; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
    th {{ background: #4a4a4a; color: #fff; text-align: left; padding: 10px; }}
    td {{ padding: 8px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }}
    tr:nth-child(even) {{ background: #f9f9f9; }}
    .score {{ font-weight: bold; padding: 3px 9px; border-radius: 3px; display: inline-block; font-size: 13px; }}
    .score-high {{ background: #c8e6c9; color: #2e7d32; }}
    .score-medium {{ background: #fff9c4; color: #f57f17; }}
    .score-low {{ background: #ffcdd2; color: #c62828; }}
    .score-col {{ width: 90px; }}
    .criteria {{ width: 20%; }}
    .output pre {{ background: #f5f5f5; border: 1px solid #ddd; padding: 8px; border-radius: 3px; overflow: auto; white-space: pre-wrap; font-size: 12px; margin: 0; }}
    ul {{ margin: 6px 0; padding-left: 20px; }}
    li {{ margin-bottom: 4px; }}
  </style>
</head>
<body>

<div class="header">
  <h1 style="margin:0 0 4px">Prompt Evaluation Report</h1>
  <div style="font-size:13px;color:#666;margin-bottom:12px">{task_description}</div>
  <div class="summary-stats">
    <div class="stat-box">
      <div>Overall Score</div>
      <div class="stat-value">{score_badge(overall)}</div>
    </div>
    <div class="stat-box">
      <div>Structural</div>
      <div class="stat-value">{score_badge(structural)}</div>
    </div>
    <div class="stat-box">
      <div>Output Quality</div>
      <div class="stat-value">{score_badge(output_score)}</div>
    </div>
    {'<div class="stat-box"><div>vs Baseline</div><div class="stat-value">' + delta_html + "</div></div>" if delta_html else ""}
    <div class="stat-box">
      <div>Pass Rate (≥7)</div>
      <div class="stat-value">{pass_rate:.0f}%</div>
    </div>
    <div class="stat-box">
      <div>Test Cases</div>
      <div class="stat-value">{total_tests}{cached_badge}</div>
    </div>
  </div>
</div>

<h2>Prompt Evaluated</h2>
<div class="prompt-box">{prompt_escaped}</div>

<h2>Structural Analysis</h2>
<table>
  <thead><tr><th>Dimension</th><th>Score</th><th>Notes</th></tr></thead>
  <tbody>{dim_rows}</tbody>
</table>

<h2>Output-Based Test Results</h2>
{'<p style="color:#888;font-style:italic">No test results available.</p>' if not test_rows else f'<table><thead><tr><th>Scenario</th><th style="width:20%">Solution Criteria</th><th>Output</th><th class="score-col">Score</th><th style="width:20%">Reasoning</th></tr></thead><tbody>{test_rows}</tbody></table>'}

<h2>Synthesis</h2>
<div style="display:flex;gap:24px;flex-wrap:wrap">
  <div style="flex:1;min-width:200px">
    <strong>Strengths</strong>
    <ul>{strengths_html}</ul>
  </div>
  <div style="flex:1;min-width:200px">
    <strong>Weaknesses</strong>
    <ul>{weaknesses_html}</ul>
  </div>
  <div style="flex:1;min-width:200px">
    <strong>Suggestions</strong>
    <ul>{suggestions_html}</ul>
  </div>
</div>

<h2>Improved Prompt</h2>
<div class="prompt-box">{improved_escaped}</div>

</body>
</html>"""
