import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('panda_nuts.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.dtypes)

# Compute efficiency = nuts_opened per second
df['efficiency'] = df['nuts_opened'] / df['seconds']
print("\nEfficiency stats:")
print(df['efficiency'].describe())

# Encode sex and help
df['sex_bin'] = (df['sex'] == 'm').astype(int)
df['help_bin'] = (df['help'].str.lower() == 'y').astype(int)

# --- Age vs efficiency ---
r_age, p_age = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nAge-efficiency: r={r_age:.3f}, p={p_age:.4f}")

# --- Sex vs efficiency ---
male_eff = df[df['sex_bin'] == 1]['efficiency']
female_eff = df[df['sex_bin'] == 0]['efficiency']
t_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"Sex t-test: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"Male mean={male_eff.mean():.4f}, Female mean={female_eff.mean():.4f}")

# --- Help vs efficiency ---
help_y = df[df['help_bin'] == 1]['efficiency']
help_n = df[df['help_bin'] == 0]['efficiency']
t_help, p_help = stats.ttest_ind(help_y, help_n)
print(f"Help t-test: t={t_help:.3f}, p={p_help:.4f}")
print(f"Help=Y mean={help_y.mean():.4f}, Help=N mean={help_n.mean():.4f}")

# --- OLS regression ---
X = df[['age', 'sex_bin', 'help_bin']].copy()
X = sm.add_constant(X)
y = df['efficiency']
model = sm.OLS(y, X).fit()
print("\nOLS summary:")
print(model.summary())

# --- Decision ---
# Determine if any of the three predictors is significant (p < 0.05)
sig_age = p_age < 0.05
sig_sex = p_sex < 0.05
sig_help = p_help < 0.05

ols_params = model.pvalues
print("\nOLS p-values:")
print(ols_params)

sig_age_ols = ols_params.get('age', 1.0) < 0.05
sig_sex_ols = ols_params.get('sex_bin', 1.0) < 0.05
sig_help_ols = ols_params.get('help_bin', 1.0) < 0.05

# Score based on how many predictors are significant and direction of effects
n_sig = sum([sig_age_ols, sig_sex_ols, sig_help_ols])

# Build explanation
parts = []
parts.append(f"Age-efficiency correlation: r={r_age:.3f}, p={p_age:.4f} ({'significant' if sig_age else 'not significant'}).")
parts.append(f"Sex difference: male_eff={male_eff.mean():.4f}, female_eff={female_eff.mean():.4f}, p={p_sex:.4f} ({'significant' if sig_sex else 'not significant'}).")
parts.append(f"Help effect: help_yes={help_y.mean():.4f}, help_no={help_n.mean():.4f}, p={p_help:.4f} ({'significant' if sig_help else 'not significant'}).")
parts.append(f"OLS: {n_sig}/3 predictors significant at p<0.05.")

# Scoring: if at least 2 of 3 are significant -> high score; 1 -> medium; 0 -> low
if n_sig >= 2:
    response = 80
elif n_sig == 1:
    response = 55
else:
    # Check univariate significance
    n_sig_uni = sum([sig_age, sig_sex, sig_help])
    if n_sig_uni >= 2:
        response = 65
    elif n_sig_uni == 1:
        response = 40
    else:
        response = 20

explanation = " ".join(parts) + f" Overall response score: {response}."
print("\nExplanation:", explanation)

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclustion.txt written.")
