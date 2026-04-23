import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from agentic_imodels import SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor

# Load the research question from info.json
with open('info.json', 'r') as f:
    info = json.load(f)
research_question = info['research_questions'][0]

print("="*80)
print("RESEARCH QUESTION:")
print(research_question)
print("="*80)
print()

# Load the dataset
df = pd.read_csv('hurricane.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn types:")
print(df.dtypes)
print()

# Summary statistics
print("="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(df.describe())
print()

# Key variables for the research question:
# DV: alldeaths (fatalities - proxy for precautionary measures)
# IV: masfem (femininity of hurricane name)
# Controls: category, min, ndam (or ndam15), wind, year/elapsedyrs

# Check for missing values
print("Missing values:")
print(df.isnull().sum())
print()

# Drop rows with missing values in key columns
df_clean = df.dropna(subset=['masfem', 'alldeaths', 'category', 'min', 'wind', 'ndam', 'elapsedyrs'])
print(f"After dropping missing values: {df_clean.shape[0]} rows (from {df.shape[0]})")
print()

# Use the cleaned dataframe
df = df_clean

# Bivariate analysis: Correlation between masfem and alldeaths
print("="*80)
print("BIVARIATE ANALYSIS")
print("="*80)
corr_masfem_deaths, pval_corr = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"Pearson correlation (masfem vs alldeaths): r={corr_masfem_deaths:.4f}, p={pval_corr:.4f}")

# Also check spearman (more robust to outliers)
spearman_corr, spearman_pval = stats.spearmanr(df['masfem'], df['alldeaths'])
print(f"Spearman correlation (masfem vs alldeaths): rho={spearman_corr:.4f}, p={spearman_pval:.4f}")
print()

# Distribution of key variables
print("Distribution of alldeaths:")
print(df['alldeaths'].describe())
print(f"Skewness: {df['alldeaths'].skew():.2f}")
print()

print("Distribution of masfem:")
print(df['masfem'].describe())
print()

# Classical regression analysis with statsmodels
print("="*80)
print("CLASSICAL STATISTICAL TESTS (OLS)")
print("="*80)

# Model 1: Bivariate (masfem only)
X1 = sm.add_constant(df[['masfem']])
y = df['alldeaths']
model1 = sm.OLS(y, X1).fit()
print("\nModel 1: Bivariate regression (alldeaths ~ masfem)")
print(model1.summary())
print()

# Model 2: With hurricane severity controls
# Controls: category, min (pressure), wind, ndam (damage)
control_cols = ['category', 'min', 'wind', 'ndam']
X2 = sm.add_constant(df[['masfem'] + control_cols])
model2 = sm.OLS(y, X2).fit()
print("\nModel 2: Controlled regression (alldeaths ~ masfem + controls)")
print(model2.summary())
print()

# Model 3: With additional temporal control
control_cols_full = ['category', 'min', 'wind', 'ndam', 'elapsedyrs']
X3 = sm.add_constant(df[['masfem'] + control_cols_full])
model3 = sm.OLS(y, X3).fit()
print("\nModel 3: Fully controlled regression (alldeaths ~ masfem + all controls)")
print(model3.summary())
print()

# Since alldeaths has outliers, let's also try log transformation
df['log_deaths'] = np.log1p(df['alldeaths'])
X_log = sm.add_constant(df[['masfem'] + control_cols_full])
model_log = sm.OLS(df['log_deaths'], X_log).fit()
print("\nModel 4: Log-transformed DV (log(alldeaths+1) ~ masfem + controls)")
print(model_log.summary())
print()

# Interpretable models
print("="*80)
print("INTERPRETABLE MODELS (agentic_imodels)")
print("="*80)

# Prepare feature matrix for interpretable models
feature_cols = ['masfem', 'category', 'min', 'wind', 'ndam', 'elapsedyrs']
X = df[feature_cols]
y = df['alldeaths']

print(f"\nFeatures used: {feature_cols}")
print(f"Target: alldeaths")
print()

# Fit SmartAdditiveRegressor (honest GAM)
print("="*80)
print("Model 1: SmartAdditiveRegressor (honest GAM)")
print("="*80)
sar = SmartAdditiveRegressor()
sar.fit(X, y)
print(sar)
print()

# Fit HingeEBMRegressor (high-rank model)
print("="*80)
print("Model 2: HingeEBMRegressor (display-predict decoupled)")
print("="*80)
hebm = HingeEBMRegressor()
hebm.fit(X, y)
print(hebm)
print()

# Fit WinsorizedSparseOLSRegressor (honest sparse linear)
print("="*80)
print("Model 3: WinsorizedSparseOLSRegressor (honest sparse linear)")
print("="*80)
wsols = WinsorizedSparseOLSRegressor()
wsols.fit(X, y)
print(wsols)
print()

# Interpretation and conclusion
print("="*80)
print("INTERPRETATION AND CONCLUSION")
print("="*80)
print()

print("Research Question:")
print(research_question)
print()

print("Key Findings:")
print()
print("1. BIVARIATE RELATIONSHIP:")
print(f"   - Pearson correlation (masfem vs alldeaths): r={corr_masfem_deaths:.4f}, p={pval_corr:.4f}")
print(f"   - Spearman correlation: rho={spearman_corr:.4f}, p={spearman_pval:.4f}")
if pval_corr < 0.05:
    print(f"   - The bivariate correlation is statistically significant (p<0.05)")
    if corr_masfem_deaths > 0:
        print(f"   - Direction: POSITIVE (more feminine names → more deaths)")
else:
    print(f"   - The bivariate correlation is NOT statistically significant (p≥0.05)")
print()

print("2. CONTROLLED REGRESSION (OLS with severity controls):")
masfem_coef_model2 = model2.params['masfem']
masfem_pval_model2 = model2.pvalues['masfem']
print(f"   - Coefficient (with category, min, wind, ndam): β={masfem_coef_model2:.3f}, p={masfem_pval_model2:.4f}")

masfem_coef_model3 = model3.params['masfem']
masfem_pval_model3 = model3.pvalues['masfem']
print(f"   - Coefficient (with all controls): β={masfem_coef_model3:.3f}, p={masfem_pval_model3:.4f}")

masfem_coef_log = model_log.params['masfem']
masfem_pval_log = model_log.pvalues['masfem']
print(f"   - Coefficient (log-transformed DV): β={masfem_coef_log:.3f}, p={masfem_pval_log:.4f}")
print()

print("3. INTERPRETABLE MODEL INSIGHTS:")
print("   - The SmartAdditiveRegressor, HingeEBMRegressor, and WinsorizedSparseOLSRegressor")
print("     all show the contribution of 'masfem' relative to hurricane severity variables.")
print("   - Review the printed model forms above to assess:")
print("     * Whether 'masfem' was RETAINED (non-zero coefficient/effect)")
print("     * The DIRECTION of the effect (positive or negative)")
print("     * The MAGNITUDE/RANK relative to other predictors")
print("     * The SHAPE of the relationship (linear, threshold, etc.)")
print()

# Decision logic for conclusion
explanation_parts = []

# Check bivariate evidence
if pval_corr < 0.05 and corr_masfem_deaths > 0:
    bivariate_evidence = "POSITIVE"
    explanation_parts.append(f"Bivariate analysis shows a significant positive correlation (r={corr_masfem_deaths:.3f}, p={pval_corr:.4f}).")
elif pval_corr < 0.05 and corr_masfem_deaths < 0:
    bivariate_evidence = "NEGATIVE"
    explanation_parts.append(f"Bivariate analysis shows a significant negative correlation (r={corr_masfem_deaths:.3f}, p={pval_corr:.4f}).")
else:
    bivariate_evidence = "WEAK"
    explanation_parts.append(f"Bivariate correlation is not statistically significant (r={corr_masfem_deaths:.3f}, p={pval_corr:.4f}).")

# Check controlled regression evidence
if masfem_pval_model3 < 0.05:
    if masfem_coef_model3 > 0:
        controlled_evidence = "POSITIVE"
        explanation_parts.append(f"After controlling for hurricane severity (category, pressure, wind, damage) and temporal factors, masfem retains a significant POSITIVE effect (β={masfem_coef_model3:.2f}, p={masfem_pval_model3:.4f}).")
    else:
        controlled_evidence = "NEGATIVE"
        explanation_parts.append(f"After controlling for hurricane severity, masfem shows a significant NEGATIVE effect (β={masfem_coef_model3:.2f}, p={masfem_pval_model3:.4f}).")
else:
    controlled_evidence = "WEAK"
    explanation_parts.append(f"After controlling for hurricane severity, masfem is NOT statistically significant (β={masfem_coef_model3:.2f}, p={masfem_pval_model3:.4f}).")

# Interpretable models provide shape/robustness evidence
explanation_parts.append("Interpretable models (SmartAdditiveRegressor, HingeEBMRegressor, WinsorizedSparseOLSRegressor) characterize the effect shape and importance. Review the printed model forms to confirm direction, magnitude, and whether masfem was retained or zeroed out by sparse/hinge models.")

# Calibrate Likert score based on evidence
if bivariate_evidence == "POSITIVE" and controlled_evidence == "POSITIVE":
    # Strong evidence: bivariate AND controlled both significant and positive
    score = 75
    explanation_parts.append("CONCLUSION: Strong evidence that more feminine hurricane names are associated with more deaths, supporting the hypothesis that feminine names lead to fewer precautionary measures and hence more fatalities.")
elif bivariate_evidence == "POSITIVE" and controlled_evidence == "WEAK":
    # Moderate evidence: bivariate significant but effect disappears with controls
    score = 30
    explanation_parts.append("CONCLUSION: The bivariate relationship suggests more feminine names → more deaths, but this effect becomes non-significant after controlling for hurricane severity. The relationship may be confounded by severity or spurious.")
elif bivariate_evidence == "WEAK" and controlled_evidence == "POSITIVE":
    # Rare case: controlled shows effect but bivariate doesn't
    score = 50
    explanation_parts.append("CONCLUSION: Controlling for confounders reveals a positive effect of name femininity on deaths, though the bivariate relationship is weak. Moderate evidence for the hypothesis.")
elif bivariate_evidence == "WEAK" and controlled_evidence == "WEAK":
    # No evidence
    score = 10
    explanation_parts.append("CONCLUSION: No significant relationship found between name femininity and deaths, either in bivariate or controlled analyses. Little to no support for the hypothesis.")
elif bivariate_evidence == "NEGATIVE" or controlled_evidence == "NEGATIVE":
    # Evidence against the hypothesis
    score = 5
    explanation_parts.append("CONCLUSION: Evidence shows a NEGATIVE relationship (opposite of the hypothesis), suggesting more feminine names are associated with fewer, not more, deaths.")
else:
    # Default/unclear
    score = 25
    explanation_parts.append("CONCLUSION: The evidence is mixed or unclear. Weak to no support for the hypothesis.")

explanation = " ".join(explanation_parts)

print("="*80)
print("FINAL CONCLUSION")
print("="*80)
print(f"Likert Score (0=strong No, 100=strong Yes): {score}")
print()
print("Explanation:")
print(explanation)
print()

# Write conclusion to file
conclusion = {
    "response": score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("Conclusion written to conclusion.txt")
