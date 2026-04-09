#!/usr/bin/env python3
"""
Script to analyze hurricane dataset and test research question from info.json.
"""
import json
import pandas as pd
from scipy.stats import pearsonr

def main():
    # Read research question
    with open('info.json') as f:
        info = json.load(f)
    question = info.get('research_questions', [None])[0]

    # Load hurricane data
    df = pd.read_csv('hurricane.csv')

    # Perform Pearson correlation between femininity index and fatalities
    masfem = df['masfem']
    deaths = df['alldeaths']
    r, p = pearsonr(masfem, deaths)

    # Determine significance at alpha=0.05
    alpha = 0.05
    significant = p < alpha
    # Positive relationship supports hypothesis
    response = 100 if significant and r > 0 else 0

    # Construct explanation
    if significant:
        explanation = (
            f"Pearson correlation between name femininity (masfem) and fatalities (alldeaths) "
            f"is {r:.3f} (p={p:.3e}), indicating a significant "
            f"{'positive' if r > 0 else 'negative'} relationship."
        )
    else:
        explanation = (
            f"Pearson correlation between name femininity (masfem) and fatalities (alldeaths) "
            f"is {r:.3f} (p={p:.3e}), indicating no significant relationship."
        )

    # Write conclusion as JSON
    result = {"response": response, "explanation": explanation}
    with open('conclusion.txt', 'w') as f:
        json.dump(result, f)


if __name__ == '__main__':
    main()
