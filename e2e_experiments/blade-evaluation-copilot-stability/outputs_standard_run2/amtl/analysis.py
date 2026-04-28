import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the dataset
df = pd.read_csv('amtl.csv')

print("="*80)
print("DATASET EXPLORATION")
print("="*80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nGenus distribution:")
print(df['genus'].value_counts())

print(f"\nBasic statistics:")
print(df.describe())

# Calculate AMTL rate (proportion of teeth lost)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print(f"\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].agg(['mean', 'std', 'count']))

print(f"\nAMTL rate summary by genus:")
for genus in df['genus'].unique():
    genus_data = df[df['genus'] == genus]
    print(f"\n{genus}:")
    print(f"  Mean AMTL rate: {genus_data['amtl_rate'].mean():.4f}")
    print(f"  Median AMTL rate: {genus_data['amtl_rate'].median():.4f}")
    print(f"  Mean age: {genus_data['age'].mean():.2f}")
    print(f"  N specimens: {len(genus_data['specimen'].unique())}")

print("\n" + "="*80)
print("STATISTICAL ANALYSIS")
print("="*80)

# Create binary indicator for Homo sapiens
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Encode categorical variables for regression
df['tooth_class_encoded'] = pd.Categorical(df['tooth_class']).codes

# Simple comparison: Homo sapiens vs all others
human_amtl = df[df['genus'] == 'Homo sapiens']['amtl_rate']
nonhuman_amtl = df[df['genus'] != 'Homo sapiens']['amtl_rate']

print(f"\nSimple comparison (without controls):")
print(f"Homo sapiens mean AMTL rate: {human_amtl.mean():.4f}")
print(f"Non-human primates mean AMTL rate: {nonhuman_amtl.mean():.4f}")
print(f"Difference: {human_amtl.mean() - nonhuman_amtl.mean():.4f}")

# T-test
t_stat, p_value = stats.ttest_ind(human_amtl, nonhuman_amtl)
print(f"\nT-test (Homo sapiens vs others):")
print(f"  t-statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.6f}")

# Mann-Whitney U test (non-parametric alternative)
u_stat, u_pvalue = stats.mannwhitneyu(human_amtl, nonhuman_amtl, alternative='two-sided')
print(f"\nMann-Whitney U test:")
print(f"  U-statistic: {u_stat:.4f}")
print(f"  p-value: {u_pvalue:.6f}")

print("\n" + "="*80)
print("REGRESSION ANALYSIS (CONTROLLING FOR CONFOUNDERS)")
print("="*80)

# For binomial regression with proportions, we need to use statsmodels
# We'll use weighted least squares or GLM with binomial family

# Create dummy variables for tooth class
df = pd.get_dummies(df, columns=['tooth_class'], prefix='tooth', drop_first=True)

# GLM with binomial family (logistic regression for proportions)
# We need to specify the number of trials (sockets) and successes (num_amtl)
print("\nBinomial GLM Model (controlling for age, sex, tooth class):")

# Prepare formula for statsmodels
# Need to create binary outcome variable for each tooth socket
# Alternative: use proportion with weights

# Let's use a simpler approach: Linear regression with AMTL rate as outcome
# First, create centered/standardized versions of continuous predictors
df['age_std'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['prob_male_std'] = (df['prob_male'] - df['prob_male'].mean()) / df['prob_male'].std()

# OLS regression
tooth_cols = [col for col in df.columns if col.startswith('tooth_')]
formula = f"amtl_rate ~ is_human + age_std + prob_male_std + {' + '.join(tooth_cols)}"

model = smf.ols(formula=formula, data=df).fit()
print(model.summary())

print(f"\n\nKey findings:")
print(f"Coefficient for is_human (Homo sapiens): {model.params['is_human']:.6f}")
print(f"P-value for is_human: {model.pvalues['is_human']:.6f}")
print(f"95% CI for is_human: [{model.conf_int().loc['is_human', 0]:.6f}, {model.conf_int().loc['is_human', 1]:.6f}]")

# Alternative: Binomial GLM (more appropriate for count data)
print("\n" + "="*80)
print("BINOMIAL GLM (more appropriate for count data)")
print("="*80)

# Create a dataset where each row represents a single observation
# For GLM with binomial family, we model num_amtl with sockets as weights
glm_formula = f"num_amtl ~ is_human + age_std + prob_male_std + {' + '.join(tooth_cols)}"

# Use GLM with binomial family
# For proportions, we need to provide the binomial denominator
# We'll use the alternative approach: expand the data

# Actually, let's use the aggregate approach with weights
# Create success/failure format for binomial GLM
df['num_no_amtl'] = df['sockets'] - df['num_amtl']

# Stack the data to create binary outcomes
expanded_data = []
for idx, row in df.iterrows():
    # Add rows for AMTL (success)
    for _ in range(int(row['num_amtl'])):
        expanded_data.append({
            'outcome': 1,
            'is_human': row['is_human'],
            'age_std': row['age_std'],
            'prob_male_std': row['prob_male_std'],
            **{col: row[col] for col in tooth_cols}
        })
    # Add rows for no AMTL (failure)
    for _ in range(int(row['num_no_amtl'])):
        expanded_data.append({
            'outcome': 0,
            'is_human': row['is_human'],
            'age_std': row['age_std'],
            'prob_male_std': row['prob_male_std'],
            **{col: row[col] for col in tooth_cols}
        })

expanded_df = pd.DataFrame(expanded_data)

# Logistic regression on expanded data
logit_formula = f"outcome ~ is_human + age_std + prob_male_std + {' + '.join(tooth_cols)}"
logit_model = smf.logit(formula=logit_formula, data=expanded_df).fit(disp=False)

print(logit_model.summary())

print(f"\n\nKey findings from Binomial GLM:")
print(f"Coefficient for is_human (Homo sapiens): {logit_model.params['is_human']:.6f}")
print(f"Odds ratio: {np.exp(logit_model.params['is_human']):.4f}")
print(f"P-value for is_human: {logit_model.pvalues['is_human']:.6f}")
print(f"95% CI for is_human coefficient: [{logit_model.conf_int().loc['is_human', 0]:.6f}, {logit_model.conf_int().loc['is_human', 1]:.6f}]")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Determine the answer based on statistical significance
ols_pvalue = model.pvalues['is_human']
glm_pvalue = logit_model.pvalues['is_human']
ols_coef = model.params['is_human']
glm_coef = logit_model.params['is_human']

print(f"\nOLS model (linear regression on AMTL rate):")
print(f"  - Coefficient: {ols_coef:.6f}")
print(f"  - P-value: {ols_pvalue:.6f}")
print(f"  - Interpretation: {'Significant' if ols_pvalue < 0.05 else 'Not significant'} at α=0.05")

print(f"\nGLM model (binomial/logistic regression):")
print(f"  - Coefficient: {glm_coef:.6f}")
print(f"  - Odds ratio: {np.exp(glm_coef):.4f}")
print(f"  - P-value: {glm_pvalue:.6f}")
print(f"  - Interpretation: {'Significant' if glm_pvalue < 0.05 else 'Not significant'} at α=0.05")

# Determine response score
if ols_pvalue < 0.001 and glm_pvalue < 0.001:
    if ols_coef > 0 and glm_coef > 0:
        response_score = 95
        explanation = (f"Strong evidence that Homo sapiens have higher AMTL rates than non-human primates "
                      f"after controlling for age, sex, and tooth class. Both OLS (p={ols_pvalue:.6f}) and "
                      f"binomial GLM (p={glm_pvalue:.6f}) show highly significant positive effects. "
                      f"The GLM odds ratio of {np.exp(glm_coef):.2f} indicates humans have "
                      f"{(np.exp(glm_coef)-1)*100:.1f}% higher odds of AMTL.")
    else:
        response_score = 5
        explanation = (f"Strong evidence that Homo sapiens have LOWER AMTL rates than non-human primates "
                      f"after controlling for age, sex, and tooth class. Both models show highly significant "
                      f"negative effects (OLS p={ols_pvalue:.6f}, GLM p={glm_pvalue:.6f}).")
elif ols_pvalue < 0.05 or glm_pvalue < 0.05:
    # At least one model shows significance
    if ols_coef > 0 or glm_coef > 0:
        response_score = 70
        explanation = (f"Moderate evidence that Homo sapiens have higher AMTL rates than non-human primates "
                      f"after controlling for age, sex, and tooth class. "
                      f"OLS p={ols_pvalue:.6f}, GLM p={glm_pvalue:.6f}. "
                      f"Results are statistically significant in at least one model.")
    else:
        response_score = 30
        explanation = (f"Some evidence against the hypothesis - models suggest Homo sapiens may have "
                      f"lower or similar AMTL rates. OLS p={ols_pvalue:.6f}, GLM p={glm_pvalue:.6f}.")
else:
    # Neither model shows significance
    response_score = 20
    explanation = (f"No significant evidence that Homo sapiens have higher AMTL rates than non-human primates "
                  f"after controlling for age, sex, and tooth class. OLS p={ols_pvalue:.6f}, GLM p={glm_pvalue:.6f}. "
                  f"The relationship is not statistically significant at α=0.05.")

print(f"\n\nFINAL ANSWER:")
print(f"Response score (0-100): {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Conclusion written to conclusion.txt")
print("="*80)
