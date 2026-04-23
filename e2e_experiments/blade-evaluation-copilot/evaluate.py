"""Evaluate Codex results against Blade ground-truth using LLM-as-a-judge.

For each dataset, loads the Codex conclusion and the human annotations,
then uses Azure OpenAI (keyless via Entra ID) to judge correctness,
completeness, and clarity on a 1-10 scale.

Usage:
    python evaluate.py --mode standard         # evaluate standard tools run
    python evaluate.py --mode custom_v2        # evaluate custom tools run
    python evaluate.py --output-dir outputs_standard_run1  # evaluate a specific directory
    python evaluate.py --mode standard --verbose

Authentication:
    Uses keyless Azure OpenAI via ChainedTokenCredential (AzureCli -> ManagedIdentity).
    No environment variables needed — just `az login` locally or run on a managed identity host.
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from openai import AzureOpenAI
from azure.identity import ChainedTokenCredential, AzureCliCredential, ManagedIdentityCredential, get_bearer_token_provider

scope = "https://cognitiveservices.azure.com/.default"
credential = get_bearer_token_provider(ChainedTokenCredential(
    AzureCliCredential(), # first check local
    ManagedIdentityCredential(), # then check managed identity (for cluster jobs)
), scope)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_BLADE_DIR = os.path.join(
    SCRIPT_DIR, "..", "example-blade-repo", "blade_bench", "datasets"
)
_SIBLING_DIR = os.path.join(
    SCRIPT_DIR, "..", "blade-evaluation", "outputs_standard_run1"
)
if os.path.isdir(_BLADE_DIR):
    DATASETS_DIR = _BLADE_DIR
elif os.path.isdir(_SIBLING_DIR):
    DATASETS_DIR = _SIBLING_DIR
else:
    DATASETS_DIR = os.path.join(SCRIPT_DIR, "outputs")

DATASETS = [
    "affairs", "amtl", "boxes", "caschools", "crofoot", "fertility",
    "fish", "hurricane", "mortgage", "panda_nuts", "reading", "soccer",
    "teachingratings",
]

JUDGE_PROMPT = """You are an expert data science evaluator. You are given:
1. A research question about a dataset
2. A description of the dataset
3. A summary of human expert annotations (ground truth model specifications)
4. An AI agent's conclusion (a 0-100 Likert score and explanation)

Evaluate the AI agent's analysis on three dimensions. Use the FULL 1-10 scale — do not cluster scores around 7-8. A score of 5 means mediocre, 7 means good, 9-10 means excellent.

**Correctness** (1-10): Does the agent reach a well-supported, defensible conclusion?

A correct conclusion is not just "right direction" — it must be WELL-JUSTIFIED by the evidence. Reward analyses where:
- The Likert score is well-calibrated: proportional to the strength of evidence, not just binary significant/not
- The conclusion is validated across multiple modeling approaches (e.g., confirmed by both regression and interpretable models)
- The agent distinguishes between bivariate associations and effects that survive controlling for confounders
- Feature importance or effect sizes are used to calibrate confidence (e.g., a variable that is "significant" but ranks last in importance should get a lower score than one that is significant AND the top predictor)
- The agent correctly identifies when a variable has NO effect (backed by evidence like zero coefficients, low importance rankings, or Lasso exclusion — not just a high p-value)

An answer that says "p < 0.05 so Yes" is less correct than one that says "p < 0.05 in OLS AND rank 1 in importance at 32% AND confirmed by a second interpretable model, so strong Yes."

Score 1-3: Wrong conclusion or fundamentally flawed reasoning
Score 4-5: Right direction but poorly justified or miscalibrated Likert score
Score 6-7: Correct conclusion with basic justification (significance tests only)
Score 8-9: Correct conclusion validated by multiple approaches with calibrated confidence
Score 10: Expert-level — correct conclusion backed by convergent evidence from multiple models, with well-calibrated Likert score reflecting actual effect strength

**Completeness** (1-10): Does the analysis go beyond basic tests to deeply understand the data?

A complete analysis doesn't just test for significance — it explains HOW variables relate. Reward analyses that:
- Include control variables and confounders in the analysis
- Go beyond p-values to describe the shape of relationships (e.g., linear vs nonlinear, thresholds, diminishing effects)
- Report feature importance or effect sizes to show WHICH variables matter most
- Use interpretable models to reveal the direction and magnitude of each feature's effect
- Identify whether effects are linear, have thresholds, or show nonlinear patterns
- Compare findings across multiple modeling approaches for robustness

The agent does NOT need to match every single expert annotation. What matters is depth of understanding.

Score 1-3: Only basic bivariate tests, ignores confounders
Score 4-5: Some controls but no deeper investigation of effect shapes or importance
Score 6-7: Controls included, some investigation of effect sizes or feature importance
Score 8-9: Thorough analysis with feature importance, effect shapes, and multiple approaches
Score 10: Comprehensive — reports effect shapes, importance rankings, nonlinear patterns, and confounders

**Clarity** (1-10): Is the explanation clear, well-structured, and rich in interpretable insight?

A great explanation doesn't just report p-values — it helps the reader UNDERSTAND the data relationships. Reward explanations that:
- Report the **direction** of each important feature's effect (positive, negative, nonlinear)
- Quantify **relative importance** — which features matter most and by how much (e.g., "hours is the 2nd most important predictor at 19.8% importance, behind livebait at 45.2%")
- Describe the **shape** of relationships — are effects linear, or are there thresholds/nonlinear patterns? (e.g., "age has a threshold effect: no effect below 25, strong positive effect above 25")
- Show **robustness** — do findings hold across different models or approaches?
- Clearly connect statistical evidence to the final conclusion
- Explain not just WHETHER a relationship exists, but HOW it works

An explanation that only says "p < 0.05 therefore significant" is shallow.
An explanation that says "income has a positive linear effect (coef=0.52, rank 1 in importance at 32%), while age shows a nonlinear threshold at 30 (importance=18%, rank 2), and gender has no effect (zeroed out by Lasso)" is rich and insightful.

Penalize:
- Contradictions between the Likert score and the evidence
- Listing statistics without interpreting what they mean for the research question
- Missing information about which features matter most or how they relate to the outcome

Score 1-3: Confusing or contradictory
Score 4-5: Only reports significance, no insight into relationships
Score 6-7: Reports direction and some effect sizes, but limited depth
Score 8-9: Reports importance rankings, effect directions, and relationship shapes across features
Score 10: Exceptionally insightful — quantifies relative importance, describes effect shapes (linear/nonlinear/threshold), shows robustness across models, and clearly connects all evidence to the conclusion

Respond ONLY with a JSON object:
{{"correctness": <int>, "completeness": <int>, "clarity": <int>, "explanation": "<brief justification>"}}

---

**Research Question:** {research_question}

**Dataset Description:** {dataset_description}

**Human Expert Annotations Summary:**
{annotations_summary}

**AI Agent Conclusion:**
- Likert Score (0=strong No, 100=strong Yes): {response}
- Explanation: {agent_explanation}
"""


def get_client() -> AzureOpenAI:
    """Create Azure OpenAI client using Entra ID token."""
    return AzureOpenAI(
        api_version="2025-04-01-preview",
        azure_endpoint="https://dl-openai-3.openai.azure.com/",
        azure_ad_token_provider=credential,
    )


def load_conclusion(dataset: str, output_dir: str) -> dict | None:
    """Load the Codex-generated conclusion.txt."""
    path = os.path.join(output_dir, dataset, "conclusion.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        with open(path) as f:
            text = f.read().strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        return {"raw": text, "parse_error": str(e)}


def load_research_question(dataset: str) -> str:
    """Load the research question from info.json."""
    path = os.path.join(DATASETS_DIR, dataset, "info.json")
    with open(path) as f:
        info = json.load(f)
    return info["research_questions"][0]


def load_dataset_description(dataset: str) -> str:
    """Load dataset description from info.json."""
    path = os.path.join(DATASETS_DIR, dataset, "info.json")
    with open(path) as f:
        info = json.load(f)
    desc = info["data_desc"]
    summary = desc.get("dataset_description", "")
    fields = []
    for f_info in desc.get("fields", [])[:10]:
        col = f_info["column"]
        props = f_info.get("properties", {})
        field_desc = props.get("description", "")
        fields.append(f"  - {col}: {field_desc}" if field_desc else f"  - {col}")
    return f"{summary}\n\nFields:\n" + "\n".join(fields)


def load_annotations_summary(dataset: str) -> str:
    """Build a text summary of human expert annotations."""
    path = os.path.join(DATASETS_DIR, dataset, "annotations.csv")
    if not os.path.exists(path):
        return "No human annotations available."

    df = pd.read_csv(path)
    lines = [f"Total expert specifications: {len(df)}"]

    concepts = []
    for _, row in df.iterrows():
        if pd.notna(row.get("conceptual_spec_json")):
            try:
                spec = json.loads(row["conceptual_spec_json"])
                concept_desc = spec.get("concept", "")
                var_type = spec.get("variable_type", "")
                if concept_desc:
                    concepts.append(f"{var_type}: {concept_desc}")
            except (json.JSONDecodeError, TypeError):
                pass
    if concepts:
        lines.append(f"Conceptual variables identified by experts:")
        for c in concepts[:8]:
            lines.append(f"  - {c}")

    model_count = 0
    model_types = set()
    for _, row in df.iterrows():
        if pd.notna(row.get("model_spec_json")):
            try:
                spec = json.loads(row["model_spec_json"])
                model_count += 1
                code = str(spec)
                for mtype in ["OLS", "GLM", "Poisson", "NegativeBinomial",
                              "logit", "probit", "lmer", "lm(", "glm("]:
                    if mtype.lower() in code.lower():
                        model_types.add(mtype)
            except (json.JSONDecodeError, TypeError):
                pass
    if model_count:
        lines.append(f"Model specifications: {model_count}")
        if model_types:
            lines.append(f"Model types used: {', '.join(sorted(model_types))}")

    transform_count = df["annotate_transform_spec_json"].notna().sum()
    if transform_count:
        lines.append(f"Data transformation specifications: {transform_count}")

    return "\n".join(lines)


def judge_dataset(
    client: AzureOpenAI,
    deployment: str,
    dataset: str,
    output_dir: str,
) -> dict:
    """Use LLM-as-a-judge to evaluate a single dataset's Codex output."""
    conclusion = load_conclusion(dataset, output_dir)
    if conclusion is None:
        return {"dataset": dataset, "status": "missing", "error": "no conclusion.txt"}
    if "parse_error" in conclusion:
        return {
            "dataset": dataset,
            "status": "parse_error",
            "error": conclusion.get("parse_error", ""),
            "raw": conclusion.get("raw", "")[:200],
        }

    question = load_research_question(dataset)
    desc = load_dataset_description(dataset)
    annotations = load_annotations_summary(dataset)

    response_val = conclusion.get("response", "N/A")
    explanation = conclusion.get("explanation", "No explanation provided.")

    prompt = JUDGE_PROMPT.format(
        research_question=question,
        dataset_description=desc,
        annotations_summary=annotations,
        response=response_val,
        agent_explanation=explanation,
    )

    try:
        response = client.responses.create(
            model=deployment,
            input=prompt,
        )
        raw = response.output_text.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            scores = json.loads(raw[start : end + 1])
        else:
            scores = {"parse_error": raw}
    except Exception as e:
        return {"dataset": dataset, "status": "api_error", "error": str(e)}

    return {
        "dataset": dataset,
        "status": "ok",
        "agent_response": response_val,
        "correctness": scores.get("correctness"),
        "completeness": scores.get("completeness"),
        "clarity": scores.get("clarity"),
        "judge_explanation": scores.get("explanation", ""),
    }


def evaluate_all(
    datasets: list[str] | None = None,
    verbose: bool = False,
    output_dir: str = None,
    results_path: str = None,
):
    """Run LLM-as-a-judge evaluation on all datasets."""
    client = get_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    datasets = datasets or DATASETS

    results = []
    for dataset in datasets:
        print(f"Evaluating: {dataset} ({os.path.basename(output_dir)})...", end=" ", flush=True)
        result = judge_dataset(client, deployment, dataset, output_dir)
        results.append(result)

        if result["status"] == "ok":
            c = result["correctness"]
            comp = result["completeness"]
            cl = result["clarity"]
            print(f"correctness={c} completeness={comp} clarity={cl}")
            if verbose:
                print(f"  Judge: {result['judge_explanation']}")
        else:
            print(f"[{result['status']}] {result.get('error', '')[:80]}")

        time.sleep(0.5)  # Rate limiting

    # Summary
    df = pd.DataFrame(results)
    ok = df[df["status"] == "ok"]

    if len(ok) > 0:
        print(f"\n{'Dataset':20s} {'Response':>8s} {'Correct':>8s} {'Complete':>8s} {'Clarity':>8s}")
        print("-" * 56)
        for _, row in ok.iterrows():
            print(
                f"{row['dataset']:20s} {str(row['agent_response']):>8s} "
                f"{row['correctness']:>8d} {row['completeness']:>8d} {row['clarity']:>8d}"
            )
        print("-" * 56)
        print(
            f"{'AVERAGE':20s} {'':>8s} "
            f"{ok['correctness'].mean():>8.2f} {ok['completeness'].mean():>8.2f} "
            f"{ok['clarity'].mean():>8.2f}"
        )
        overall = (ok["correctness"].mean() + ok["completeness"].mean() + ok["clarity"].mean()) / 3
        print(f"\nOverall average score: {overall:.2f} / 10.00")

    # Save results
    if results_path is None:
        results_path = os.path.join(SCRIPT_DIR, "judge_results", f"results_{os.path.basename(output_dir)}.csv")
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Blade Codex results with LLM-as-a-judge")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show judge explanations")
    parser.add_argument("--mode", type=str, choices=["standard", "custom_v2"], default="standard",
                        help="Which run to evaluate: 'standard' or 'custom_v2'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Explicit output directory to evaluate (overrides --mode)")
    parser.add_argument("--results-path", type=str, default=None,
                        help="Explicit path for results CSV")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(SCRIPT_DIR, output_dir)
    else:
        output_dir = os.path.join(SCRIPT_DIR, f"outputs_{args.mode}")

    ds = [args.dataset] if args.dataset else None
    evaluate_all(datasets=ds, verbose=args.verbose, output_dir=output_dir,
                 results_path=args.results_path)
