import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('amtl.csv')
print("Shape:", df.shape)
print(df['genus'].value_counts())

# Compute AMTL rate per row
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Summary by genus
print("\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].agg(['mean', 'std', 'count']))

# Binary: is specimen Homo sapiens?
df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# Encode tooth_class as dummies
df['tooth_anterior'] = (df['tooth_class'] == 'Anterior').astype(int)
df['tooth_posterior'] = (df['tooth_class'] == 'Posterior').astype(int)

# Proper binomial GLM: endog is [successes, failures]
endog = np.column_stack([df['num_amtl'], df['sockets'] - df['num_amtl']])
X = sm.add_constant(df[['is_homo', 'age', 'prob_male', 'tooth_anterior', 'tooth_posterior']])

model = sm.GLM(endog, X, family=sm.families.Binomial())
result = model.fit()
print("\nGLM Binomial Results:")
print(result.summary())

homo_coef = result.params['is_homo']
homo_pval = result.pvalues['is_homo']
homo_ci = result.conf_int().loc['is_homo']

print(f"\nHomo sapiens coefficient: {homo_coef:.4f}, p-value: {homo_pval:.4e}")
print(f"95% CI: [{homo_ci[0]:.4f}, {homo_ci[1]:.4f}]")
print(f"Odds ratio: {np.exp(homo_coef):.4f}")

# Raw comparison
homo_rates = df[df['genus'] == 'Homo sapiens']['amtl_rate']
nonhomo_rates = df[df['genus'] != 'Homo sapiens']['amtl_rate']
tstat, tpval = stats.ttest_ind(homo_rates, nonhomo_rates)
print(f"\nRaw t-test: Homo mean={homo_rates.mean():.4f}, Non-Homo mean={nonhomo_rates.mean():.4f}")
print(f"t={tstat:.4f}, p={tpval:.4e}")

# ANOVA across all genera
groups = [df[df['genus'] == g]['amtl_rate'].values for g in df['genus'].unique()]
fstat, fpval = stats.f_oneway(*groups)
print(f"\nANOVA across genera: F={fstat:.4f}, p={fpval:.4e}")

is_significant = homo_pval < 0.05
is_positive = homo_coef > 0

if is_significant and is_positive:
    score = 90
    explanation = (
        f"The GLM binomial regression (proper 2-column response) shows Homo sapiens have significantly higher "
        f"AMTL rates after controlling for age, sex, and tooth class. "
        f"Homo coefficient={homo_coef:.4f} (OR={np.exp(homo_coef):.4f}), p={homo_pval:.2e}, "
        f"95% CI [{homo_ci[0]:.4f}, {homo_ci[1]:.4f}]. "
        f"Raw means: Homo={homo_rates.mean():.4f} vs Non-Homo={nonhomo_rates.mean():.4f} "
        f"(t-test p={tpval:.2e}, ANOVA p={fpval:.2e}). "
        f"Strong evidence that modern humans have higher AMTL frequencies than non-human primates."
    )
elif is_significant and not is_positive:
    score = 10
    explanation = (
        f"GLM shows Homo sapiens have significantly LOWER AMTL rates after controlling for covariates "
        f"(coef={homo_coef:.4f}, OR={np.exp(homo_coef):.4f}, p={homo_pval:.2e}). "
        f"Raw means: Homo={homo_rates.mean():.4f} vs Non-Homo={nonhomo_rates.mean():.4f}."
    )
else:
    score = 30
    explanation = (
        f"No significant difference in AMTL after controlling for age, sex, tooth class "
        f"(coef={homo_coef:.4f}, p={homo_pval:.2e})."
    )

conclusion = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\nconclusion.txt written.")
print(json.dumps(conclusion, indent=2))
