import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import json

# Load the data
df = pd.read_csv('reading.csv')

print("=" * 80)
print("RESEARCH QUESTION: Does 'Reader View' improve reading speed for individuals with dyslexia?")
print("=" * 80)
print()

# Explore the data
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Focus on key variables
print("\n" + "=" * 80)
print("KEY VARIABLES ANALYSIS")
print("=" * 80)

# Check dyslexia distribution
print("\nDyslexia Distribution:")
print(df['dyslexia'].value_counts().sort_index())
print("\nDyslexia Binary Distribution:")
print(df['dyslexia_bin'].value_counts().sort_index())

# Check reader_view distribution
print("\nReader View Distribution:")
print(df['reader_view'].value_counts().sort_index())

# Speed variable analysis
print("\nSpeed Statistics:")
print(df['speed'].describe())

# Remove outliers in speed (extreme values that might be errors)
print("\nRemoving extreme outliers in speed (> 99th percentile)...")
speed_99 = df['speed'].quantile(0.99)
df_clean = df[df['speed'] <= speed_99].copy()
print(f"Removed {len(df) - len(df_clean)} rows with speed > {speed_99:.2f}")
print(f"New dataset size: {len(df_clean)} rows")

print("\n" + "=" * 80)
print("ANALYSIS: Reader View Effect on Reading Speed for Dyslexic Individuals")
print("=" * 80)

# Separate data by dyslexia status
dyslexic_data = df_clean[df_clean['dyslexia_bin'] == 1].copy()
non_dyslexic_data = df_clean[df_clean['dyslexia_bin'] == 0].copy()

print(f"\nDyslexic individuals: {len(dyslexic_data)} observations")
print(f"Non-dyslexic individuals: {len(non_dyslexic_data)} observations")

# For dyslexic individuals: compare speed with and without reader view
dyslexic_with_rv = dyslexic_data[dyslexic_data['reader_view'] == 1]['speed']
dyslexic_without_rv = dyslexic_data[dyslexic_data['reader_view'] == 0]['speed']

print("\n" + "-" * 80)
print("DYSLEXIC INDIVIDUALS:")
print("-" * 80)
print(f"With Reader View: n={len(dyslexic_with_rv)}, mean={dyslexic_with_rv.mean():.2f}, std={dyslexic_with_rv.std():.2f}")
print(f"Without Reader View: n={len(dyslexic_without_rv)}, mean={dyslexic_without_rv.mean():.2f}, std={dyslexic_without_rv.std():.2f}")
print(f"Difference in mean speed: {dyslexic_with_rv.mean() - dyslexic_without_rv.mean():.2f}")

# T-test for dyslexic individuals
if len(dyslexic_with_rv) > 0 and len(dyslexic_without_rv) > 0:
    t_stat_dys, p_value_dys = stats.ttest_ind(dyslexic_with_rv, dyslexic_without_rv)
    print(f"\nTwo-sample t-test for dyslexic individuals:")
    print(f"  t-statistic: {t_stat_dys:.4f}")
    print(f"  p-value: {p_value_dys:.4f}")
    
    if p_value_dys < 0.05:
        if dyslexic_with_rv.mean() > dyslexic_without_rv.mean():
            print(f"  Result: SIGNIFICANT POSITIVE EFFECT (p < 0.05)")
            print(f"  Reader View INCREASES reading speed for dyslexic individuals")
        else:
            print(f"  Result: SIGNIFICANT NEGATIVE EFFECT (p < 0.05)")
            print(f"  Reader View DECREASES reading speed for dyslexic individuals")
    else:
        print(f"  Result: NO SIGNIFICANT EFFECT (p >= 0.05)")

# For comparison: non-dyslexic individuals
non_dyslexic_with_rv = non_dyslexic_data[non_dyslexic_data['reader_view'] == 1]['speed']
non_dyslexic_without_rv = non_dyslexic_data[non_dyslexic_data['reader_view'] == 0]['speed']

print("\n" + "-" * 80)
print("NON-DYSLEXIC INDIVIDUALS (for comparison):")
print("-" * 80)
print(f"With Reader View: n={len(non_dyslexic_with_rv)}, mean={non_dyslexic_with_rv.mean():.2f}, std={non_dyslexic_with_rv.std():.2f}")
print(f"Without Reader View: n={len(non_dyslexic_without_rv)}, mean={non_dyslexic_without_rv.mean():.2f}, std={non_dyslexic_without_rv.std():.2f}")
print(f"Difference in mean speed: {non_dyslexic_with_rv.mean() - non_dyslexic_without_rv.mean():.2f}")

if len(non_dyslexic_with_rv) > 0 and len(non_dyslexic_without_rv) > 0:
    t_stat_non, p_value_non = stats.ttest_ind(non_dyslexic_with_rv, non_dyslexic_without_rv)
    print(f"\nTwo-sample t-test for non-dyslexic individuals:")
    print(f"  t-statistic: {t_stat_non:.4f}")
    print(f"  p-value: {p_value_non:.4f}")

# Regression analysis with interaction term
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS WITH INTERACTION TERM")
print("=" * 80)

# Create interaction term and drop rows with missing dyslexia_bin
df_reg = df_clean.dropna(subset=['dyslexia_bin']).copy()
df_reg['dyslexia_x_readerview'] = df_reg['dyslexia_bin'] * df_reg['reader_view']

print(f"\nUsing {len(df_reg)} observations for regression (after removing missing dyslexia values)")

# Prepare data for regression
X = df_reg[['reader_view', 'dyslexia_bin', 'dyslexia_x_readerview']].copy()
X = sm.add_constant(X)
y = df_reg['speed']

# Fit OLS model
model = sm.OLS(y, X).fit()
print("\nRegression Results:")
print(model.summary())

print("\n" + "-" * 80)
print("INTERPRETATION OF REGRESSION COEFFICIENTS:")
print("-" * 80)
print(f"Intercept: {model.params['const']:.2f} (baseline speed: no dyslexia, no reader view)")
print(f"reader_view: {model.params['reader_view']:.2f} (effect of reader view for non-dyslexic)")
print(f"dyslexia_bin: {model.params['dyslexia_bin']:.2f} (effect of dyslexia without reader view)")
print(f"dyslexia_x_readerview: {model.params['dyslexia_x_readerview']:.2f} (INTERACTION: additional effect of reader view for dyslexic)")
print(f"\np-value for interaction term: {model.pvalues['dyslexia_x_readerview']:.4f}")

# Calculate the total effect of reader view for dyslexic individuals
total_effect_dyslexic = model.params['reader_view'] + model.params['dyslexia_x_readerview']
print(f"\nTotal effect of reader view for dyslexic individuals: {total_effect_dyslexic:.2f}")
print(f"  (reader_view coefficient + interaction coefficient)")

# Final conclusion
print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

# Determine response score and explanation
if len(dyslexic_with_rv) == 0 or len(dyslexic_without_rv) == 0:
    response_score = 50
    explanation = "Insufficient data to determine if Reader View improves reading speed for dyslexic individuals."
else:
    # Use the p-value from the t-test for dyslexic individuals
    if p_value_dys < 0.001:
        # Very strong evidence
        if dyslexic_with_rv.mean() > dyslexic_without_rv.mean():
            response_score = 95
            explanation = f"Strong evidence (p={p_value_dys:.4f}) that Reader View improves reading speed for dyslexic individuals. Mean speed increased from {dyslexic_without_rv.mean():.1f} to {dyslexic_with_rv.mean():.1f} words/min with Reader View."
        else:
            response_score = 5
            explanation = f"Strong evidence (p={p_value_dys:.4f}) that Reader View does NOT improve reading speed for dyslexic individuals. Speed decreased from {dyslexic_without_rv.mean():.1f} to {dyslexic_with_rv.mean():.1f} words/min."
    elif p_value_dys < 0.01:
        # Strong evidence
        if dyslexic_with_rv.mean() > dyslexic_without_rv.mean():
            response_score = 85
            explanation = f"Strong evidence (p={p_value_dys:.4f}) that Reader View improves reading speed for dyslexic individuals. Mean speed increased from {dyslexic_without_rv.mean():.1f} to {dyslexic_with_rv.mean():.1f} words/min."
        else:
            response_score = 15
            explanation = f"Strong evidence (p={p_value_dys:.4f}) that Reader View does NOT improve reading speed for dyslexic individuals. Speed decreased."
    elif p_value_dys < 0.05:
        # Moderate evidence
        if dyslexic_with_rv.mean() > dyslexic_without_rv.mean():
            response_score = 70
            explanation = f"Moderate evidence (p={p_value_dys:.4f}) that Reader View improves reading speed for dyslexic individuals. Mean speed increased from {dyslexic_without_rv.mean():.1f} to {dyslexic_with_rv.mean():.1f} words/min."
        else:
            response_score = 30
            explanation = f"Moderate evidence (p={p_value_dys:.4f}) that Reader View does NOT improve reading speed for dyslexic individuals."
    else:
        # No significant effect
        if dyslexic_with_rv.mean() > dyslexic_without_rv.mean():
            response_score = 40
            explanation = f"No significant evidence (p={p_value_dys:.4f}) that Reader View improves reading speed for dyslexic individuals. Observed difference in means could be due to chance."
        else:
            response_score = 40
            explanation = f"No significant evidence (p={p_value_dys:.4f}) that Reader View affects reading speed for dyslexic individuals."

print(f"\nResponse Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete! Results written to conclusion.txt")
print("=" * 80)
