import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('hurricane.csv')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nHead:\n", df.head())
print("\nDescribe:\n", df.describe())

# Research question: Do hurricanes with more feminine names lead to more deaths
# (because people perceive them as less threatening and take fewer precautions)?

print("\n=== Summary Statistics by Gender ===")
print(df.groupby('gender_mf')[['alldeaths', 'masfem', 'category', 'wind', 'min']].mean())

# Check correlation between femininity and deaths
corr_masfem, pval_masfem = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"\nCorrelation masfem vs alldeaths: r={corr_masfem:.4f}, p={pval_masfem:.4f}")

corr_mturk, pval_mturk = stats.pearsonr(df['masfem_mturk'], df['alldeaths'])
print(f"Correlation masfem_mturk vs alldeaths: r={corr_mturk:.4f}, p={pval_mturk:.4f}")

# T-test: female vs male hurricanes on deaths
female_deaths = df[df['gender_mf'] == 1]['alldeaths']
male_deaths = df[df['gender_mf'] == 0]['alldeaths']
t_stat, t_pval = stats.ttest_ind(female_deaths, male_deaths)
print(f"\nT-test (female vs male deaths): t={t_stat:.4f}, p={t_pval:.4f}")
print(f"Female mean deaths: {female_deaths.mean():.2f}, Male mean deaths: {male_deaths.mean():.2f}")

# OLS regression: deaths ~ masfem (unadjusted)
X_simple = sm.add_constant(df['masfem'])
model_simple = sm.OLS(df['alldeaths'], X_simple).fit()
print("\n=== OLS: alldeaths ~ masfem (unadjusted) ===")
print(model_simple.summary().tables[1])

# OLS regression controlling for storm severity
controls = ['masfem', 'category', 'wind', 'min', 'ndam']
df_clean = df[controls + ['alldeaths']].dropna()
X_full = sm.add_constant(df_clean[controls])
model_full = sm.OLS(df_clean['alldeaths'], X_full).fit()
print("\n=== OLS: alldeaths ~ masfem + severity controls ===")
print(model_full.summary().tables[1])

# Log-transform deaths (heavily skewed)
df['log_deaths'] = np.log1p(df['alldeaths'])
df_clean2 = df[controls + ['log_deaths']].dropna()
X_log = sm.add_constant(df_clean2[controls])
model_log = sm.OLS(df_clean2['log_deaths'], X_log).fit()
print("\n=== OLS: log(alldeaths+1) ~ masfem + severity controls ===")
print(model_log.summary().tables[1])

# Check with masfem_mturk
controls_mturk = ['masfem_mturk', 'category', 'wind', 'min', 'ndam']
df_clean3 = df[controls_mturk + ['log_deaths']].dropna()
X_mturk = sm.add_constant(df_clean3[controls_mturk])
model_mturk = sm.OLS(df_clean3['log_deaths'], X_mturk).fit()
print("\n=== OLS: log(alldeaths+1) ~ masfem_mturk + severity controls ===")
print(model_mturk.summary().tables[1])

# Use EBM for interpretable nonlinear analysis
try:
    from interpret.glassbox import ExplainableBoostingRegressor
    features = ['masfem', 'category', 'wind', 'min', 'ndam']
    df_ebm = df[features + ['log_deaths']].dropna()
    X_ebm = df_ebm[features].values
    y_ebm = df_ebm['log_deaths'].values
    ebm = ExplainableBoostingRegressor(random_state=42)
    ebm.fit(X_ebm, y_ebm)
    importances = dict(zip(features, ebm.term_importances()))
    print("\n=== EBM Feature Importances ===")
    for k, v in sorted(importances.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:.4f}")
except Exception as e:
    print(f"EBM failed: {e}")

# Spearman correlation (robust to outliers)
spear_r, spear_p = stats.spearmanr(df['masfem'], df['alldeaths'])
print(f"\nSpearman correlation masfem vs alldeaths: r={spear_r:.4f}, p={spear_p:.4f}")

# Summary of findings
print("\n=== FINDINGS SUMMARY ===")
print(f"Simple correlation (masfem vs deaths): r={corr_masfem:.4f}, p={pval_masfem:.4f}")
print(f"Spearman (robust): r={spear_r:.4f}, p={spear_p:.4f}")
print(f"T-test (female vs male): p={t_pval:.4f}")
print(f"OLS unadjusted p-value for masfem: {model_simple.pvalues['masfem']:.4f}")
print(f"OLS adjusted p-value for masfem: {model_full.pvalues['masfem']:.4f}")
print(f"OLS log-adjusted p-value for masfem: {model_log.pvalues['masfem']:.4f}")

# Determine response score
# The original study (Jung et al. 2014) claimed an effect but was heavily criticized.
# Simonsohn et al. (specification curve) showed the result is not robust.
# Let's base our answer on the data.

masfem_pval_adjusted = model_log.pvalues['masfem']
masfem_coef_adjusted = model_log.params['masfem']

print(f"\nKey result: masfem coef (log-adjusted) = {masfem_coef_adjusted:.4f}, p = {masfem_pval_adjusted:.4f}")

if masfem_pval_adjusted < 0.05 and masfem_coef_adjusted > 0:
    response = 70
    explanation = (
        f"The analysis finds a statistically significant positive relationship between hurricane name femininity "
        f"(masfem) and deaths after controlling for storm severity (coef={masfem_coef_adjusted:.3f}, "
        f"p={masfem_pval_adjusted:.3f}), supporting the claim that more feminine hurricane names are associated "
        f"with higher death tolls, consistent with the hypothesis that they are perceived as less threatening."
    )
elif masfem_pval_adjusted < 0.10 and masfem_coef_adjusted > 0:
    response = 45
    explanation = (
        f"The analysis finds a marginally significant positive relationship between hurricane name femininity and deaths "
        f"when controlling for severity (coef={masfem_coef_adjusted:.3f}, p={masfem_pval_adjusted:.3f}). "
        f"The evidence is weak and the result is not robust across specifications, "
        f"consistent with Simonsohn et al.'s specification curve showing the finding is fragile."
    )
else:
    response = 25
    explanation = (
        f"The analysis does not find a statistically significant relationship between hurricane name femininity and deaths "
        f"after controlling for storm severity (coef={masfem_coef_adjusted:.3f}, p={masfem_pval_adjusted:.3f}). "
        f"Simple correlation is r={corr_masfem:.3f} (p={pval_masfem:.3f}). "
        f"The hypothesis that more feminine hurricane names lead to fewer precautionary measures and more deaths "
        f"is not supported by this data in a robust, statistically significant way. "
        f"This is consistent with the Simonsohn et al. specification curve analysis showing the original finding lacks robustness."
    )

import json
conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
