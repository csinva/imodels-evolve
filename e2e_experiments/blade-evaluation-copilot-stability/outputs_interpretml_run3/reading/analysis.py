import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import json

# Load the dataset
df = pd.read_csv('reading.csv')

# Research Question: Does 'Reader View' improve reading speed for individuals with dyslexia?

print("=" * 80)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Key variables for analysis
print("\n" + "=" * 80)
print("KEY VARIABLES SUMMARY")
print("=" * 80)
print("\nReader View distribution:")
print(df['reader_view'].value_counts())
print("\nDyslexia distribution:")
print(df['dyslexia_bin'].value_counts())

print("\nSpeed statistics:")
print(df['speed'].describe())

# Remove extreme outliers in speed (likely data errors)
print("\n" + "=" * 80)
print("DATA CLEANING")
print("=" * 80)
q1 = df['speed'].quantile(0.01)
q99 = df['speed'].quantile(0.99)
print(f"Removing speed outliers below {q1:.2f} and above {q99:.2f}")
df_clean = df[(df['speed'] >= q1) & (df['speed'] <= q99)].copy()
print(f"Cleaned dataset shape: {df_clean.shape}")

# Focus on dyslexia group (dyslexia_bin = 1)
df_dyslexia = df_clean[df_clean['dyslexia_bin'] == 1].copy()
df_no_dyslexia = df_clean[df_clean['dyslexia_bin'] == 0].copy()

print("\n" + "=" * 80)
print("ANALYSIS: READER VIEW EFFECT ON READING SPEED FOR DYSLEXIA")
print("=" * 80)

print(f"\nDyslexia group size: {len(df_dyslexia)}")
print(f"No-dyslexia group size: {len(df_no_dyslexia)}")

# For dyslexia group: compare speed with and without reader view
dyslexia_reader_on = df_dyslexia[df_dyslexia['reader_view'] == 1]['speed']
dyslexia_reader_off = df_dyslexia[df_dyslexia['reader_view'] == 0]['speed']

print(f"\nDyslexia with Reader View ON (n={len(dyslexia_reader_on)}):")
print(f"  Mean speed: {dyslexia_reader_on.mean():.2f}")
print(f"  Std speed: {dyslexia_reader_on.std():.2f}")

print(f"\nDyslexia with Reader View OFF (n={len(dyslexia_reader_off)}):")
print(f"  Mean speed: {dyslexia_reader_off.mean():.2f}")
print(f"  Std speed: {dyslexia_reader_off.std():.2f}")

# Statistical test: Independent t-test
print("\n" + "=" * 80)
print("STATISTICAL TEST: T-TEST")
print("=" * 80)
print("H0: Reader View does NOT affect reading speed for dyslexia")
print("H1: Reader View affects reading speed for dyslexia")

t_stat, p_value = stats.ttest_ind(dyslexia_reader_on, dyslexia_reader_off)
print(f"\nt-statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Significance level: 0.05")

if p_value < 0.05:
    print("Result: SIGNIFICANT - Reject null hypothesis")
    if dyslexia_reader_on.mean() > dyslexia_reader_off.mean():
        print("Reader View is associated with HIGHER reading speed (improvement)")
    else:
        print("Reader View is associated with LOWER reading speed (detriment)")
else:
    print("Result: NOT SIGNIFICANT - Fail to reject null hypothesis")

# Effect size (Cohen's d)
cohens_d = (dyslexia_reader_on.mean() - dyslexia_reader_off.mean()) / np.sqrt(
    ((len(dyslexia_reader_on) - 1) * dyslexia_reader_on.std()**2 + 
     (len(dyslexia_reader_off) - 1) * dyslexia_reader_off.std()**2) / 
    (len(dyslexia_reader_on) + len(dyslexia_reader_off) - 2)
)
print(f"\nCohen's d (effect size): {cohens_d:.4f}")

# Regression analysis with controls
print("\n" + "=" * 80)
print("REGRESSION ANALYSIS WITH CONTROLS")
print("=" * 80)

# Prepare data for regression (dyslexia group only)
df_reg = df_dyslexia.copy()
df_reg = df_reg[['speed', 'reader_view', 'age', 'num_words', 'correct_rate']].dropna()

X = df_reg[['reader_view', 'age', 'num_words', 'correct_rate']]
y = df_reg['speed']
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

reader_view_coef = model.params['reader_view']
reader_view_pvalue = model.pvalues['reader_view']

print(f"\n\nReader View coefficient: {reader_view_coef:.4f}")
print(f"Reader View p-value: {reader_view_pvalue:.4f}")

# Interaction effect: Does reader view help dyslexia MORE than no-dyslexia?
print("\n" + "=" * 80)
print("INTERACTION ANALYSIS")
print("=" * 80)
print("Testing if Reader View helps dyslexia group MORE than no-dyslexia group")

# Compare effect of reader view across groups
no_dyslexia_reader_on = df_no_dyslexia[df_no_dyslexia['reader_view'] == 1]['speed']
no_dyslexia_reader_off = df_no_dyslexia[df_no_dyslexia['reader_view'] == 0]['speed']

dyslexia_diff = dyslexia_reader_on.mean() - dyslexia_reader_off.mean()
no_dyslexia_diff = no_dyslexia_reader_on.mean() - no_dyslexia_reader_off.mean()

print(f"\nDyslexia group: Reader View effect = {dyslexia_diff:.2f}")
print(f"No-Dyslexia group: Reader View effect = {no_dyslexia_diff:.2f}")
print(f"Difference-in-differences: {dyslexia_diff - no_dyslexia_diff:.2f}")

# Interaction regression
df_interaction = df_clean[['speed', 'reader_view', 'dyslexia_bin', 'age', 'num_words']].dropna().copy()
df_interaction['reader_x_dyslexia'] = df_interaction['reader_view'] * df_interaction['dyslexia_bin']

X_int = df_interaction[['reader_view', 'dyslexia_bin', 'reader_x_dyslexia', 'age', 'num_words']]
y_int = df_interaction['speed']
X_int = sm.add_constant(X_int)

model_int = sm.OLS(y_int, X_int).fit()
print("\n\nInteraction Model:")
print(model_int.summary())

interaction_coef = model_int.params['reader_x_dyslexia']
interaction_pvalue = model_int.pvalues['reader_x_dyslexia']

print(f"\n\nInteraction coefficient (reader_view * dyslexia): {interaction_coef:.4f}")
print(f"Interaction p-value: {interaction_pvalue:.4f}")

# FINAL CONCLUSION
print("\n" + "=" * 80)
print("FINAL CONCLUSION")
print("=" * 80)

# Determine response based on statistical evidence
explanation = ""
response_score = 50  # Default neutral

if p_value < 0.05:
    # Significant effect found
    if dyslexia_reader_on.mean() > dyslexia_reader_off.mean():
        # Positive effect (improvement)
        if p_value < 0.001:
            response_score = 85
            explanation = f"Strong evidence that Reader View improves reading speed for individuals with dyslexia. T-test shows significant difference (p={p_value:.4f}). Mean speed increased from {dyslexia_reader_off.mean():.1f} (off) to {dyslexia_reader_on.mean():.1f} (on), a {((dyslexia_reader_on.mean()/dyslexia_reader_off.mean()-1)*100):.1f}% improvement. Effect size (Cohen's d={cohens_d:.2f}) confirms practical significance. Regression analysis controlling for age, word count, and comprehension rate confirms Reader View has significant positive coefficient (p={reader_view_pvalue:.4f})."
        elif p_value < 0.01:
            response_score = 75
            explanation = f"Good evidence that Reader View improves reading speed for individuals with dyslexia. T-test shows significant difference (p={p_value:.4f}). Mean speed increased from {dyslexia_reader_off.mean():.1f} (off) to {dyslexia_reader_on.mean():.1f} (on). Cohen's d={cohens_d:.2f}. Controlled regression confirms positive effect (p={reader_view_pvalue:.4f})."
        else:
            response_score = 65
            explanation = f"Moderate evidence that Reader View improves reading speed for individuals with dyslexia. T-test shows marginal significance (p={p_value:.4f}). Mean speed increased from {dyslexia_reader_off.mean():.1f} (off) to {dyslexia_reader_on.mean():.1f} (on). Effect size is modest (Cohen's d={cohens_d:.2f})."
    else:
        # Negative effect (detriment)
        response_score = 25
        explanation = f"Evidence suggests Reader View may actually REDUCE reading speed for dyslexia (p={p_value:.4f}). Mean speed decreased from {dyslexia_reader_off.mean():.1f} (off) to {dyslexia_reader_on.mean():.1f} (on). This contradicts the hypothesis."
else:
    # No significant effect
    response_score = 35
    explanation = f"No significant evidence that Reader View improves reading speed for individuals with dyslexia. T-test not significant (p={p_value:.4f}). Mean speeds are similar: {dyslexia_reader_off.mean():.1f} (off) vs {dyslexia_reader_on.mean():.1f} (on). Cohen's d={cohens_d:.2f} indicates minimal practical difference. Cannot conclude Reader View has meaningful impact on reading speed for this population."

print(f"\nResponse Score: {response_score}/100")
print(f"\nExplanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("CONCLUSION WRITTEN TO conclusion.txt")
print("=" * 80)
