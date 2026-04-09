#!/usr/bin/env python3
"""
Script to analyze whether having children decreases engagement in extramarital affairs.
"""
import json
import pandas as pd
from scipy.stats import ttest_ind

# Load research question and metadata
with open('info.json', 'r') as f:
    info = json.load(f)
research_question = info['research_questions'][0]

# Load the dataset
df = pd.read_csv('affairs.csv')

# Group data by children status
group_yes = df[df['children'] == 'yes']['affairs']
group_no = df[df['children'] == 'no']['affairs']

# Compute means
mean_yes = group_yes.mean()
mean_no = group_no.mean()

# Perform two-sample t-test (unequal variance)
t_stat, p_two = ttest_ind(group_yes, group_no, equal_var=False, nan_policy='omit')
# Compute one-tailed p-value for decrease (yes < no)
if t_stat < 0:
    p_one = p_two / 2
else:
    p_one = 1 - p_two / 2

# Determine significance
alpha = 0.05
dir_text = 'lower' if mean_yes < mean_no else 'higher'
if p_one < alpha and mean_yes < mean_no:
    response = 100
    explanation = (
        f"Mean affairs for those with children ({mean_yes:.3f}) is lower than without children ({mean_no:.3f}); "
        f"t-statistic={t_stat:.3f}, one-tailed p-value={p_one:.4f}, indicating a significant decrease in affairs."
    )
else:
    response = 0
    explanation = f"Mean affairs for those with children ({mean_yes:.3f}) is {dir_text} than without children ({mean_no:.3f}); t-statistic={t_stat:.3f}, one-tailed p-value={p_one:.4f}, indicating no significant decrease in affairs."

# Write conclusion to file
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
