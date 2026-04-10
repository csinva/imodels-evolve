import json
import numpy as np
import pandas as pd
from scipy import stats


def one_sided_p_from_ttest(ttest_result, alternative='less'):
    """Convert a two-sided SciPy t-test p-value into one-sided p-value."""
    t_stat = float(ttest_result.statistic)
    p_two = float(ttest_result.pvalue)

    if alternative == 'less':
        return p_two / 2 if t_stat < 0 else 1 - p_two / 2
    if alternative == 'greater':
        return p_two / 2 if t_stat > 0 else 1 - p_two / 2
    return p_two


def main():
    # 1) Read metadata / question
    with open('info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)

    question = info.get('research_questions', [''])[0]

    # 2) Load data
    df = pd.read_csv('affairs.csv')

    # Basic exploration
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())

    # Prepare groups
    children_norm = df['children'].astype(str).str.strip().str.lower()
    with_children = df.loc[children_norm == 'yes', 'affairs'].astype(float).to_numpy()
    without_children = df.loc[children_norm == 'no', 'affairs'].astype(float).to_numpy()

    if len(with_children) == 0 or len(without_children) == 0:
        raise ValueError('Could not split data into children=yes and children=no groups.')

    # Descriptive stats
    mean_with = float(np.mean(with_children))
    mean_without = float(np.mean(without_children))
    median_with = float(np.median(with_children))
    median_without = float(np.median(without_children))
    prop_with_any = float(np.mean(with_children > 0))
    prop_without_any = float(np.mean(without_children > 0))

    # 3) Statistical tests for directional claim:
    # H1: having children decreases affairs => with_children < without_children
    alpha = 0.05

    # Non-parametric distributional test
    mw = stats.mannwhitneyu(with_children, without_children, alternative='less')
    p_mw_less = float(mw.pvalue)

    # Mean-based test
    welch = stats.ttest_ind(with_children, without_children, equal_var=False)
    p_t_less = one_sided_p_from_ttest(welch, alternative='less')

    # Binary-any-affair test
    table = np.array([
        [int(np.sum(with_children > 0)), int(np.sum(with_children == 0))],
        [int(np.sum(without_children > 0)), int(np.sum(without_children == 0))],
    ])
    fisher_less = stats.fisher_exact(table, alternative='less')
    fisher_greater = stats.fisher_exact(table, alternative='greater')
    p_fisher_less = float(fisher_less.pvalue)
    p_fisher_greater = float(fisher_greater.pvalue)
    odds_ratio = float(fisher_less.statistic)

    # Also test if there is a difference in either direction
    p_t_two = float(welch.pvalue)

    # Decision logic for Likert score
    # - Strong yes only if decrease is significant and direction is lower with children.
    # - Strong no if significant in opposite direction.
    decrease_supported = (
        (mean_with < mean_without)
        and (p_mw_less < alpha)
        and (p_t_less < alpha)
        and (p_fisher_less < alpha)
    )

    opposite_supported = (
        (mean_with > mean_without)
        and (p_t_two < alpha)
        and (p_fisher_greater < alpha)
    )

    if decrease_supported:
        response = 92
        verdict = 'Evidence supports a decrease in affairs among people with children.'
    elif opposite_supported:
        response = 3
        verdict = 'Evidence contradicts the claim: people with children report more affairs, not fewer.'
    else:
        response = 20
        verdict = 'No statistically significant evidence that children decrease affairs.'

    explanation = (
        f"Research question: {question.strip()} "
        f"Dataset has {n_rows} rows, {n_cols} columns, and {missing_total} missing values. "
        f"Mean affairs: children=yes {mean_with:.3f} vs children=no {mean_without:.3f}; "
        f"median {median_with:.3f} vs {median_without:.3f}; any-affair rate {prop_with_any:.3f} vs {prop_without_any:.3f}. "
        f"Directional tests for 'decrease' (children < no-children): Mann-Whitney p={p_mw_less:.4g}, "
        f"Welch t-test one-sided p={p_t_less:.4g}, Fisher exact one-sided p={p_fisher_less:.4g}. "
        f"Two-sided Welch p={p_t_two:.4g}; Fisher test for opposite direction (children > no-children) p={p_fisher_greater:.4g}, "
        f"odds ratio={odds_ratio:.3f}. {verdict}"
    )

    # 4) Write required output file (JSON object only)
    result = {
        'response': int(response),
        'explanation': explanation,
    }

    with open('conclusion.txt', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=True)


if __name__ == '__main__':
    main()
