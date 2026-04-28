import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('amtl.csv')
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df['genus'].value_counts())

# Compute AMTL rate per row
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Summary stats by genus
print("\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].describe())

# Create binary: is_human
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Compare Homo sapiens vs non-human primates AMTL rate
human_rates = df[df['is_human'] == 1]['amtl_rate']
nonhuman_rates = df[df['is_human'] == 0]['amtl_rate']

print(f"\nHomo sapiens mean AMTL rate: {human_rates.mean():.4f}")
print(f"Non-human primates mean AMTL rate: {nonhuman_rates.mean():.4f}")

# Simple t-test
t_stat, p_val = stats.ttest_ind(human_rates, nonhuman_rates)
print(f"\nT-test: t={t_stat:.4f}, p={p_val:.6f}")

# Logistic regression / GLM approach: binomial regression
# Response: num_amtl / sockets (proportion)
# Predictors: is_human, age, prob_male, tooth_class

# Encode tooth_class
df['tooth_class_encoded'] = LabelEncoder().fit_transform(df['tooth_class'])

# Use statsmodels GLM with binomial family (logit link)
# Response: proportions with weights (sockets)
df_model = df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']).copy()
df_model['tooth_Posterior'] = (df_model['tooth_class'] == 'Posterior').astype(int)
df_model['tooth_Premolar'] = (df_model['tooth_class'] == 'Premolar').astype(int)

# Binomial GLM: logit(amtl_rate) ~ is_human + age + prob_male + tooth class
# Use freq_weights or exposure
y = df_model[['num_amtl', 'sockets']].copy()
y['not_amtl'] = y['sockets'] - y['num_amtl']
y_endog = y[['num_amtl', 'not_amtl']].values

X = sm.add_constant(df_model[['is_human', 'age', 'prob_male', 'tooth_Posterior', 'tooth_Premolar']])

glm_model = sm.GLM(y_endog, X, family=sm.families.Binomial())
result = glm_model.fit()
print("\nGLM Binomial Results:")
print(result.summary())

is_human_coef = result.params['is_human']
is_human_pval = result.pvalues['is_human']
is_human_ci_low, is_human_ci_high = result.conf_int().loc['is_human']

print(f"\nis_human coefficient: {is_human_coef:.4f}")
print(f"is_human p-value: {is_human_pval:.6e}")
print(f"is_human 95% CI: [{is_human_ci_low:.4f}, {is_human_ci_high:.4f}]")
print(f"Odds ratio for is_human: {np.exp(is_human_coef):.4f}")

# Interpret: positive coefficient + significant p-value => Homo sapiens have higher AMTL
significant = is_human_pval < 0.05
positive_effect = is_human_coef > 0

print(f"\nSignificant? {significant}")
print(f"Positive effect (higher in humans)? {positive_effect}")

# Determine response score
if significant and positive_effect:
    # Strong yes
    response = 90
    explanation = (
        f"Binomial GLM controlling for age, sex, and tooth class shows Homo sapiens have "
        f"significantly higher AMTL rates than non-human primates. "
        f"Coefficient for is_human = {is_human_coef:.4f} (OR = {np.exp(is_human_coef):.2f}), "
        f"p = {is_human_pval:.2e}, 95% CI [{is_human_ci_low:.4f}, {is_human_ci_high:.4f}]. "
        f"Human mean AMTL rate = {human_rates.mean():.4f} vs non-human = {nonhuman_rates.mean():.4f}. "
        f"Result is highly statistically significant, supporting the conclusion that modern humans "
        f"have higher AMTL frequencies after accounting for confounders."
    )
elif significant and not positive_effect:
    response = 10
    explanation = (
        f"Binomial GLM controlling for age, sex, and tooth class shows Homo sapiens have "
        f"significantly LOWER AMTL rates than non-human primates. "
        f"Coefficient for is_human = {is_human_coef:.4f} (OR = {np.exp(is_human_coef):.2f}), "
        f"p = {is_human_pval:.2e}. This contradicts the hypothesis."
    )
else:
    response = 30
    explanation = (
        f"No significant difference found. Coefficient for is_human = {is_human_coef:.4f}, "
        f"p = {is_human_pval:.2e}. Cannot conclude humans have higher AMTL rates."
    )

conclusion = {"response": response, "explanation": explanation}
print("\nConclusion:", json.dumps(conclusion, indent=2))

with open('conclusion.txt', 'w') as f:
    f.write(json.dumps(conclusion))

print("\nconclusion.txt written successfully.")
