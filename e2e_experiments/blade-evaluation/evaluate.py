"""Evaluate Codex results against Blade ground-truth using LLM-as-a-judge.

For each dataset, loads the Codex conclusion and the human annotations,
then uses Azure OpenAI to judge correctness, completeness, and clarity.

Usage:
    python evaluate.py                         # evaluate all datasets
    python evaluate.py --dataset hurricane     # evaluate one dataset
    python evaluate.py --verbose               # show judge explanations

Environment variables:
    AZURE_OPENAI_API_KEY       - Entra ID token (from refresh_token.sh)
    AZURE_OPENAI_ENDPOINT      - Azure endpoint (default: https://dl-openai-1.openai.azure.com/)
    AZURE_OPENAI_DEPLOYMENT    - Deployment name (default: gpt-4o)
    AZURE_OPENAI_API_VERSION   - API version (default: 2024-05-01-preview)
"""

import argparse
import json
import os
import sys
import time

import pandas as pd
from openai import AzureOpenAI

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
DATASETS_DIR = os.path.join(
    SCRIPT_DIR, "..", "example-blade-repo", "blade", "blade_bench", "datasets"
)

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

Evaluate the AI agent's analysis on three dimensions, each scored 1-5:

**Correctness** (1-5): Does the AI's conclusion align with sound statistical methodology?
Does it identify the right variables, use appropriate tests/models, and reach a defensible conclusion?

**Completeness** (1-5): Does the analysis consider relevant confounders, data issues,
and alternative explanations? Does it match the breadth of the human expert annotations?

**Clarity** (1-5): Is the explanation well-structured, precise, and easy to follow?

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
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if not api_key:
        print("ERROR: AZURE_OPENAI_API_KEY not set. Run: source refresh_token.sh")
        sys.exit(1)

    endpoint = os.environ.get(
        "AZURE_OPENAI_ENDPOINT", "https://dl-openai-1.openai.azure.com/"
    )
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


def load_conclusion(dataset: str) -> dict | None:
    """Load the Codex-generated conclusion.txt."""
    path = os.path.join(OUTPUT_DIR, dataset, "conclusion.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        # Try to extract JSON from file
        with open(path) as f:
            text = f.read().strip()
        # Strip anything before first { and after last }
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
    for f_info in desc.get("fields", [])[:10]:  # Cap at 10 fields
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

    # Summarize conceptual specs (IVs, DVs, controls)
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

    # Summarize model specs
    model_count = 0
    model_types = set()
    for _, row in df.iterrows():
        if pd.notna(row.get("model_spec_json")):
            try:
                spec = json.loads(row["model_spec_json"])
                model_count += 1
                # Try to extract model type
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

    # Summarize annotated transform specs
    transform_count = df["annotate_transform_spec_json"].notna().sum()
    if transform_count:
        lines.append(f"Data transformation specifications: {transform_count}")

    return "\n".join(lines)


def judge_dataset(
    client: AzureOpenAI,
    deployment: str,
    dataset: str,
) -> dict:
    """Use LLM-as-a-judge to evaluate a single dataset's Codex output."""
    conclusion = load_conclusion(dataset)
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
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )
        raw = response.choices[0].message.content.strip()
        # Parse JSON from response
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
):
    """Run LLM-as-a-judge evaluation on all datasets."""
    client = get_client()
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    datasets = datasets or DATASETS

    results = []
    for dataset in datasets:
        print(f"Evaluating: {dataset}...", end=" ", flush=True)
        result = judge_dataset(client, deployment, dataset)
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
    print("\n" + "=" * 80)
    print("BLADE LLM-AS-A-JUDGE EVALUATION RESULTS")
    print("=" * 80)

    ok = df[df["status"] == "ok"]
    missing = df[df["status"] == "missing"]
    errors = df[df["status"].isin(["parse_error", "api_error"])]

    print(f"\nCompleted: {len(ok)}/{len(df)} datasets")
    if len(missing):
        print(f"Missing conclusions: {', '.join(missing['dataset'].tolist())}")
    if len(errors):
        print(f"Errors: {', '.join(errors['dataset'].tolist())}")

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
        print(f"\nOverall average score: {overall:.2f} / 5.00")

    # Save results
    results_path = os.path.join(SCRIPT_DIR, "results.csv")
    df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Blade Codex results with LLM-as-a-judge")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset to evaluate")
    parser.add_argument("--verbose", action="store_true", help="Show judge explanations")
    args = parser.parse_args()

    ds = [args.dataset] if args.dataset else None
    evaluate_all(datasets=ds, verbose=args.verbose)
