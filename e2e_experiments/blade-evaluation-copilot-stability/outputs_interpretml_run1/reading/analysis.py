import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('reading.csv')

# Research Question: Does 'Reader View' improves reading speed for individuals with dyslexia?

# Calculate reading speed (words per millisecond, then convert to words per minute)
# Speed appears to already be calculated in the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\n=== DATA EXPLORATION ===")
print("\nBasic statistics:")
print(df.describe())

print("\n=== RESEARCH QUESTION ANALYSIS ===")
print("Does 'Reader View' improve reading speed for individuals with dyslexia?")

# Focus on dyslexia and reader_view
print("\nDyslexia distribution:")
print(df['dyslexia_bin'].value_counts())

print("\nReader view distribution:")
print(df['reader_view'].value_counts())

print("\n=== ANALYSIS 1: Effect of Reader View on Reading Speed for Dyslexic Users ===")

# Separate dyslexic and non-dyslexic participants
dyslexic = df[df['dyslexia_bin'] == 1]
non_dyslexic = df[df['dyslexia_bin'] == 0]

print(f"\nDyslexic participants: {len(dyslexic)}")
print(f"Non-dyslexic participants: {len(non_dyslexic)}")

# For dyslexic participants, compare reading speed with and without reader view
dyslexic_with_reader = dyslexic[dyslexic['reader_view'] == 1]['speed']
dyslexic_without_reader = dyslexic[dyslexic['reader_view'] == 0]['speed']

print(f"\nDyslexic WITH reader view (n={len(dyslexic_with_reader)}): mean speed = {dyslexic_with_reader.mean():.2f}")
print(f"Dyslexic WITHOUT reader view (n={len(dyslexic_without_reader)}): mean speed = {dyslexic_without_reader.mean():.2f}")

# T-test for dyslexic participants
if len(dyslexic_with_reader) > 0 and len(dyslexic_without_reader) > 0:
    t_stat, p_value = stats.ttest_ind(dyslexic_with_reader, dyslexic_without_reader)
    print(f"\nT-test for dyslexic participants:")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((dyslexic_with_reader.std()**2 + dyslexic_without_reader.std()**2) / 2)
    cohens_d = (dyslexic_with_reader.mean() - dyslexic_without_reader.mean()) / pooled_std
    print(f"  Cohen's d (effect size): {cohens_d:.4f}")

print("\n=== ANALYSIS 2: Interaction Effect (Reader View × Dyslexia) ===")

# Compare non-dyslexic participants as well
non_dyslexic_with_reader = non_dyslexic[non_dyslexic['reader_view'] == 1]['speed']
non_dyslexic_without_reader = non_dyslexic[non_dyslexic['reader_view'] == 0]['speed']

print(f"\nNon-dyslexic WITH reader view (n={len(non_dyslexic_with_reader)}): mean speed = {non_dyslexic_with_reader.mean():.2f}")
print(f"Non-dyslexic WITHOUT reader view (n={len(non_dyslexic_without_reader)}): mean speed = {non_dyslexic_without_reader.mean():.2f}")

# Regression model with interaction term
# Prepare data
df_clean = df.dropna(subset=['speed', 'reader_view', 'dyslexia_bin'])

# Create interaction term
df_clean['interaction'] = df_clean['reader_view'] * df_clean['dyslexia_bin']

# OLS regression with statsmodels for p-values
X = df_clean[['reader_view', 'dyslexia_bin', 'interaction']]
X = sm.add_constant(X)
y = df_clean['speed']

model = sm.OLS(y, X).fit()
print("\n=== OLS Regression Results ===")
print(model.summary())

print("\n=== INTERPRETATION ===")

# Extract key coefficients and p-values
reader_view_coef = model.params['reader_view']
reader_view_pval = model.pvalues['reader_view']
interaction_coef = model.params['interaction']
interaction_pval = model.pvalues['interaction']

print(f"\nReader View main effect coefficient: {reader_view_coef:.2f} (p={reader_view_pval:.4f})")
print(f"Interaction (Reader View × Dyslexia) coefficient: {interaction_coef:.2f} (p={interaction_pval:.4f})")

# Calculate specific effect for dyslexic users
dyslexic_effect = reader_view_coef + interaction_coef
print(f"\nTotal effect of Reader View for dyslexic users: {dyslexic_effect:.2f}")
print(f"Total effect of Reader View for non-dyslexic users: {reader_view_coef:.2f}")

# Determine conclusion
significance_level = 0.05

# Check if reader view improves reading speed for dyslexic users
# Higher speed = faster reading = better
improvement_for_dyslexic = dyslexic_effect > 0
statistically_significant = (interaction_pval < significance_level) or (
    (reader_view_pval < significance_level) and (interaction_pval < 0.20)
)

print(f"\nDoes reader view improve speed for dyslexic users? {improvement_for_dyslexic}")
print(f"Is the effect statistically significant? {statistically_significant}")

# Calculate response score (0-100 Likert scale)
# Consider both effect size and statistical significance
if improvement_for_dyslexic and statistically_significant:
    # Significant positive effect
    base_score = 70
    # Add bonus for strong effect size
    if abs(dyslexic_effect) > 50:
        response_score = min(90, base_score + 15)
    elif abs(dyslexic_effect) > 20:
        response_score = base_score + 10
    else:
        response_score = base_score
elif improvement_for_dyslexic and not statistically_significant:
    # Positive trend but not significant
    response_score = 55
elif not improvement_for_dyslexic and statistically_significant:
    # Significant negative effect (reader view makes it worse)
    response_score = 20
else:
    # No clear effect
    response_score = 40

# Adjust based on interaction p-value specifically
if interaction_pval < 0.001:
    response_score = min(95, response_score + 10)
elif interaction_pval < 0.01:
    response_score = min(90, response_score + 5)
elif interaction_pval > 0.20:
    response_score = max(30, response_score - 10)

print(f"\nFinal response score: {response_score}")

# Create explanation
explanation = f"Analysis of {len(df)} reading trials shows that reader view has a coefficient of {dyslexic_effect:.1f} words/min for dyslexic users. "

if statistically_significant:
    if improvement_for_dyslexic:
        explanation += f"The interaction term has p={interaction_pval:.4f}, indicating a statistically significant improvement in reading speed for individuals with dyslexia when using reader view. "
    else:
        explanation += f"Statistical analysis (p={interaction_pval:.4f}) shows reader view does not improve speed for dyslexic users. "
else:
    explanation += f"The effect is not statistically significant (interaction p={interaction_pval:.4f}). "

# Add context from descriptive statistics
if improvement_for_dyslexic:
    pct_improvement = ((dyslexic_with_reader.mean() - dyslexic_without_reader.mean()) / dyslexic_without_reader.mean() * 100)
    explanation += f"Dyslexic users showed {abs(pct_improvement):.1f}% {'higher' if pct_improvement > 0 else 'lower'} reading speed with reader view."
else:
    explanation += "Evidence does not support improved reading speed with reader view for dyslexic individuals."

# Write conclusion to file
conclusion = {
    "response": int(response_score),
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n=== CONCLUSION WRITTEN TO conclusion.txt ===")
print(json.dumps(conclusion, indent=2))
