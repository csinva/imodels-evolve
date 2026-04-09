#!/usr/bin/env python3
import json
import pandas as pd
from scipy import stats

def main():
    # Load research question (unused in analysis but read per instructions)
    with open('info.json') as f:
        info = json.load(f)
    question = info.get('research_questions', [''])[0]

    # Load dataset
    df = pd.read_csv('fertility.csv')

    # Parse dates for cycle calculation
    df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
    df['StartDateofLastPeriod'] = pd.to_datetime(
        df['StartDateofLastPeriod'], format='%m/%d/%y'
    )

    # Compute cycle day at testing (days since last period start)
    df['CycleDay'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days
    # Adjust negative values by wrapping within reported cycle length
    df['CycleDay'] = df['CycleDay'] % df['ReportedCycleLength']

    # Compute average religiosity score from the three items
    df['AvgReligiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

    # Perform Pearson correlation test
    # Drop any missing values pairwise
    valid = df[['CycleDay', 'AvgReligiosity']].dropna()
    r, p = stats.pearsonr(valid['CycleDay'], valid['AvgReligiosity'])

    # Determine significance at alpha=0.05
    significant = p < 0.05
    response = 100 if significant else 0

    # Construct explanation
    explanation = (
        f"Analysed relationship between cycle day and religiosity. "
        f"Pearson correlation r={r:.3f}, p={p:.3f}. "
        f"{'Significant at alpha=0.05,' if significant else 'Not significant at alpha=0.05,'} "
        f"{'indicating hormonal fluctuations are associated with religiosity.' if significant else 'indicating no association.'}"
    )

    # Write conclusion to file
    conclusion = {"response": response, "explanation": explanation}
    with open('conclusion.txt', 'w') as f:
        json.dump(conclusion, f)

if __name__ == '__main__':
    main()
