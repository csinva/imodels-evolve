import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('amtl.csv')
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df['genus'].value_counts())

# Compute AMTL rate (proportion) per row
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print("\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].describe())

# Create binary human indicator
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Encode tooth class
df['tooth_class_enc'] = df['tooth_class'].map({'Anterior': 0, 'Posterior': 1, 'Premolar': 2})

print("\nMean AMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].mean())

# T-test: humans vs non-humans
human_rates = df[df['is_human'] == 1]['amtl_rate'].dropna()
nonhuman_rates = df[df['is_human'] == 0]['amtl_rate'].dropna()
tstat, pval = stats.ttest_ind(human_rates, nonhuman_rates)
print(f"\nT-test humans vs non-humans: t={tstat:.4f}, p={pval:.6f}")

# Binomial logistic regression approach: use logistic regression on binary outcome
# But first let's do OLS with controls
df_clean = df.dropna(subset=['amtl_rate', 'age', 'prob_male', 'tooth_class_enc', 'is_human'])

X = df_clean[['is_human', 'age', 'prob_male', 'tooth_class_enc']]
X = sm.add_constant(X)
y = df_clean['amtl_rate']

ols = sm.OLS(y, X).fit()
print("\nOLS regression summary:")
print(ols.summary())

# Also: binomial regression (logistic with proportion outcome)
# Use statsmodels GLM with binomial family
df_clean2 = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class_enc', 'is_human'])
df_clean2 = df_clean2[df_clean2['sockets'] > 0].copy()

# GLM binomial
glm_model = sm.GLM(
    df_clean2[['num_amtl', 'sockets']].assign(failures=lambda x: x['sockets'] - x['num_amtl'])[['num_amtl', 'failures']].values,
    sm.add_constant(df_clean2[['is_human', 'age', 'prob_male', 'tooth_class_enc']]),
    family=sm.families.Binomial()
).fit()
print("\nGLM Binomial summary:")
print(glm_model.summary())

is_human_coef = glm_model.params['is_human']
is_human_pval = glm_model.pvalues['is_human']
print(f"\nis_human coefficient: {is_human_coef:.4f}, p-value: {is_human_pval:.6e}")

# ANOVA across genera
genera = df['genus'].unique()
groups = [df[df['genus'] == g]['amtl_rate'].dropna() for g in genera]
fstat, fpval = stats.f_oneway(*groups)
print(f"\nANOVA across genera: F={fstat:.4f}, p={fpval:.6f}")

# Interpretation
human_mean = human_rates.mean()
nonhuman_mean = nonhuman_rates.mean()
print(f"\nHuman mean AMTL rate: {human_mean:.4f}")
print(f"Non-human mean AMTL rate: {nonhuman_mean:.4f}")

# Determine score
# Strong evidence if: humans have higher rate AND GLM p-value is very small
significant = is_human_pval < 0.05
direction_positive = is_human_coef > 0  # positive means humans have higher AMTL

if significant and direction_positive:
    # Strong yes
    score = 90
elif significant and not direction_positive:
    score = 10
elif not significant and direction_positive:
    score = 35
else:
    score = 15

explanation = (
    f"GLM binomial regression (controlling for age, sex, tooth class) shows "
    f"is_human coefficient={is_human_coef:.4f} (log-odds), p={is_human_pval:.4e}. "
    f"Human mean AMTL rate={human_mean:.4f} vs non-human={nonhuman_mean:.4f}. "
    f"OLS is_human coef={ols.params['is_human']:.4f}, p={ols.pvalues['is_human']:.4e}. "
    f"ANOVA p={fpval:.6f}. "
    f"{'Statistically significant positive effect' if significant and direction_positive else 'Not significant or negative effect'}."
)

result = {"response": score, "explanation": explanation}
print("\nResult:", result)

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("conclusion.txt written.")
