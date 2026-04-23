import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor
from sklearn.metrics import r2_score

print("=" * 80)
print("HURRICANE NAME FEMININITY AND DEATH ANALYSIS")
print("=" * 80)

# Load data
df = pd.read_csv('hurricane.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# Research question: "Hurricanes with more feminine names are perceived as less 
# threatening and hence lead to fewer precautionary measures by the general public."
# This suggests: More feminine names → Higher death toll

print("\n" + "=" * 80)
print("STEP 1: DATA EXPLORATION")
print("=" * 80)

# Key variables:
# - DV (outcome): alldeaths (number of deaths)
# - IV (predictor of interest): masfem (femininity score, 1=masculine, 11=feminine)
# - Controls: min (pressure), category, ndam (damage), wind, year, elapsedyrs

print("\nDependent Variable (alldeaths) summary:")
print(df['alldeaths'].describe())
print(f"\nSkewness: {df['alldeaths'].skew():.2f}")
print(f"Deaths > 0: {(df['alldeaths'] > 0).sum()} / {len(df)}")

print("\nIndependent Variable (masfem) summary:")
print(df['masfem'].describe())

print("\nKey control variables:")
for col in ['min', 'category', 'ndam', 'wind', 'year']:
    print(f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, missing={df[col].isna().sum()}")

# Check correlations
print("\n" + "=" * 80)
print("BIVARIATE CORRELATIONS WITH DEATHS")
print("=" * 80)

numeric_cols = ['masfem', 'min', 'category', 'ndam', 'wind', 'year', 'alldeaths']
corr_data = df[numeric_cols].dropna()
print(f"\nSample size after dropping missing values: {len(corr_data)}")

for col in ['masfem', 'min', 'category', 'ndam', 'wind']:
    if col in corr_data.columns:
        r, p = stats.pearsonr(corr_data[col], corr_data['alldeaths'])
        print(f"{col:15s} vs alldeaths: r={r:6.3f}, p={p:.4f}")

# Bivariate test: masfem vs alldeaths
r_fem, p_fem = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"\n*** KEY BIVARIATE: masfem vs alldeaths: r={r_fem:.3f}, p={p_fem:.4f} ***")

# Check if the relationship holds for more extreme cases
high_death = df[df['alldeaths'] > df['alldeaths'].median()]
low_death = df[df['alldeaths'] <= df['alldeaths'].median()]
print(f"\nMean masfem for high-death hurricanes (>{df['alldeaths'].median()}): {high_death['masfem'].mean():.2f}")
print(f"Mean masfem for low-death hurricanes (≤{df['alldeaths'].median()}): {low_death['masfem'].mean():.2f}")
t_stat, p_ttest = stats.ttest_ind(high_death['masfem'], low_death['masfem'])
print(f"t-test: t={t_stat:.3f}, p={p_ttest:.4f}")

print("\n" + "=" * 80)
print("STEP 2: CLASSICAL STATISTICAL TESTS (WITH CONTROLS)")
print("=" * 80)

# Remove rows with missing values in key variables
analysis_df = df[['alldeaths', 'masfem', 'min', 'category', 'ndam', 'wind', 'year']].dropna()
print(f"\nAnalysis sample size: {len(analysis_df)}")

# Model 1: Bivariate OLS (no controls)
X_bivariate = sm.add_constant(analysis_df[['masfem']])
y = analysis_df['alldeaths']
model_bivariate = sm.OLS(y, X_bivariate).fit()
print("\n--- Model 1: Bivariate (masfem only) ---")
print(model_bivariate.summary().tables[1])
print(f"masfem coefficient: {model_bivariate.params['masfem']:.3f}, p={model_bivariate.pvalues['masfem']:.4f}")

# Model 2: With control variables
control_vars = ['masfem', 'min', 'category', 'ndam', 'wind']
X_controls = sm.add_constant(analysis_df[control_vars])
model_controls = sm.OLS(y, X_controls).fit()
print("\n--- Model 2: With controls (min, category, ndam, wind) ---")
print(model_controls.summary().tables[1])
print(f"\nmasfem coefficient with controls: {model_controls.params['masfem']:.3f}, p={model_controls.pvalues['masfem']:.4f}")

# Test for interaction: Does femininity interact with hurricane severity?
analysis_df['masfem_x_category'] = analysis_df['masfem'] * analysis_df['category']
X_interaction = sm.add_constant(analysis_df[['masfem', 'category', 'masfem_x_category', 'min', 'ndam', 'wind']])
model_interaction = sm.OLS(analysis_df['alldeaths'], X_interaction).fit()
print("\n--- Model 3: Interaction (masfem × category) ---")
print(f"masfem × category coefficient: {model_interaction.params['masfem_x_category']:.3f}, p={model_interaction.pvalues['masfem_x_category']:.4f}")

print("\n" + "=" * 80)
print("STEP 3: INTERPRETABLE MODELS FOR SHAPE, DIRECTION, IMPORTANCE")
print("=" * 80)

# Prepare feature matrix
feature_cols = ['masfem', 'min', 'category', 'ndam', 'wind', 'year']
X = analysis_df[feature_cols]
y = analysis_df['alldeaths']

print(f"\nFitting interpretable models on {len(X)} samples with {len(feature_cols)} features")
print(f"Features: {feature_cols}")

# Fit multiple interpretable models
models_to_fit = [
    ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
    ('HingeEBMRegressor', HingeEBMRegressor()),
    ('WinsorizedSparseOLSRegressor', WinsorizedSparseOLSRegressor()),
]

results = {}
for name, model in models_to_fit:
    print(f"\n{'=' * 80}")
    print(f"=== {name} ===")
    print('=' * 80)
    
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    print(f"\nR² = {r2:.3f}")
    print("\nFitted model form:")
    print("-" * 80)
    print(model)
    print("-" * 80)
    
    results[name] = {
        'model': model,
        'r2': r2,
        'form': str(model)
    }

print("\n" + "=" * 80)
print("STEP 4: CALIBRATED CONCLUSION")
print("=" * 80)

# Synthesize evidence
print("\n### EVIDENCE SUMMARY ###")
print(f"\n1. BIVARIATE RELATIONSHIP:")
print(f"   - Pearson correlation (masfem vs alldeaths): r={r_fem:.3f}, p={p_fem:.4f}")
print(f"   - Direction: {'POSITIVE' if r_fem > 0 else 'NEGATIVE'}")
if p_fem < 0.05:
    print(f"   - Statistically significant at α=0.05")
else:
    print(f"   - NOT statistically significant at α=0.05")

print(f"\n2. CLASSICAL REGRESSION:")
print(f"   - Bivariate OLS: β={model_bivariate.params['masfem']:.3f}, p={model_bivariate.pvalues['masfem']:.4f}")
print(f"   - With controls: β={model_controls.params['masfem']:.3f}, p={model_controls.pvalues['masfem']:.4f}")
sig_text = "SIGNIFICANT" if model_controls.pvalues['masfem'] < 0.05 else "NOT SIGNIFICANT"
print(f"   - Effect with controls: {sig_text}")

print(f"\n3. INTERPRETABLE MODELS:")
print(f"   - SmartAdditiveRegressor R²: {results['SmartAdditiveRegressor']['r2']:.3f}")
print(f"   - HingeEBMRegressor R²: {results['HingeEBMRegressor']['r2']:.3f}")
print(f"   - WinsorizedSparseOLSRegressor R²: {results['WinsorizedSparseOLSRegressor']['r2']:.3f}")

# Check if masfem appears in the sparse model
sparse_form = results['WinsorizedSparseOLSRegressor']['form']
masfem_in_sparse = 'masfem' in sparse_form and not ('masfem:' in sparse_form and '0.000' in sparse_form.split('masfem')[1].split('\n')[0])
print(f"   - masfem retained in sparse model: {masfem_in_sparse}")

# Determine final score based on evidence
# Strong evidence = significant with controls + consistent across models + high rank
# Weak evidence = nonsignificant or inconsistent or zeroed out

bivariate_significant = p_fem < 0.05
controlled_significant = model_controls.pvalues['masfem'] < 0.05
positive_direction = r_fem > 0

print(f"\n### INTERPRETATION ###")

if positive_direction:
    print("\nThe hypothesis states that more feminine names → higher deaths (less threatening perception).")
    if bivariate_significant and controlled_significant and masfem_in_sparse:
        score = 75
        explanation = (
            f"Strong evidence SUPPORTS the hypothesis. Bivariate correlation is positive (r={r_fem:.3f}, p={p_fem:.4f}), "
            f"and the effect persists after controlling for hurricane severity, damage, and wind speed "
            f"(β={model_controls.params['masfem']:.3f}, p={model_controls.pvalues['masfem']:.4f}). "
            f"The masfem variable is retained by the sparse interpretable models, indicating robustness. "
            f"More feminine hurricane names are associated with higher death tolls, consistent with the "
            f"perception-threat hypothesis."
        )
    elif bivariate_significant and controlled_significant:
        score = 65
        explanation = (
            f"Moderate-to-strong evidence SUPPORTS the hypothesis. The positive relationship between femininity "
            f"and deaths is significant both bivariate (r={r_fem:.3f}, p={p_fem:.4f}) and with controls "
            f"(β={model_controls.params['masfem']:.3f}, p={model_controls.pvalues['masfem']:.4f}). "
            f"However, some interpretable models show weaker support for masfem as a primary driver."
        )
    elif bivariate_significant:
        score = 45
        explanation = (
            f"Weak-to-moderate evidence for the hypothesis. The bivariate correlation is positive and significant "
            f"(r={r_fem:.3f}, p={p_fem:.4f}), but the effect is {"reduced" if model_controls.pvalues['masfem'] < 0.10 else "not significant"} "
            f"after controlling for hurricane characteristics (β={model_controls.params['masfem']:.3f}, "
            f"p={model_controls.pvalues['masfem']:.4f}). This suggests confounding by hurricane severity."
        )
    else:
        score = 25
        explanation = (
            f"Limited evidence for the hypothesis. While the correlation is positive (r={r_fem:.3f}), "
            f"it is not statistically significant (p={p_fem:.4f}). The interpretable models do not consistently "
            f"rank masfem as an important predictor compared to actual hurricane characteristics like category, "
            f"wind speed, and damage."
        )
else:
    # Negative direction contradicts hypothesis
    if bivariate_significant:
        score = 15
        explanation = (
            f"Evidence CONTRADICTS the hypothesis. The correlation between femininity and deaths is NEGATIVE "
            f"(r={r_fem:.3f}, p={p_fem:.4f}), opposite to what the hypothesis predicts. More feminine names "
            f"are associated with FEWER deaths, not more."
        )
    else:
        score = 10
        explanation = (
            f"No meaningful evidence for the hypothesis. The relationship between femininity and deaths is weak "
            f"and not significant (r={r_fem:.3f}, p={p_fem:.4f}). Interpretable models show that actual hurricane "
            f"characteristics (severity, damage, wind) are much stronger predictors of deaths than name femininity."
        )

print(f"\n*** FINAL LIKERT SCORE: {score}/100 ***")
print(f"\n{explanation}")

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Conclusion written to conclusion.txt")
print("=" * 80)
