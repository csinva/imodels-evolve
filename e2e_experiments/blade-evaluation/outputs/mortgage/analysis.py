#!/usr/bin/env python3
"""
analysis.py
Loads mortgage data, tests relationship between gender and mortgage approval,
and writes conclusion.txt
"""
import json
import pandas as pd
from scipy.stats import chi2_contingency

def main():
    # Load research question
    with open('info.json', 'r') as f:
        info = json.load(f)
    question = info.get('research_questions', [None])[0]

    # Load dataset
    df = pd.read_csv('mortgage.csv')

    # Contingency table: rows=female (0=male,1=female), cols=accept (0=deny,1=accept)
    table = pd.crosstab(df['female'], df['accept'])
    chi2, p, dof, expected = chi2_contingency(table)

    # Compute acceptance rates by gender
    rate_female = df.loc[df['female'] == 1, 'accept'].mean()
    rate_male = df.loc[df['female'] == 0, 'accept'].mean()

    # Determine response on Likert scale
    if p < 0.05:
        response = 100
        explanation = (
            f"Chi-squared test indicates a significant relationship between gender and mortgage approval "
            f"(p={p:.4f}). Acceptance rate for females: {rate_female:.3f}; males: {rate_male:.3f}."
        )
    else:
        response = 0
        explanation = (
            f"No significant relationship detected between gender and mortgage approval "
            f"(p={p:.4f}). Acceptance rate for females: {rate_female:.3f}; males: {rate_male:.3f}."
        )

    # Write conclusion file
    result = {"response": response, "explanation": explanation}
    with open('conclusion.txt', 'w') as f:
        json.dump(result, f)

if __name__ == '__main__':
    main()
