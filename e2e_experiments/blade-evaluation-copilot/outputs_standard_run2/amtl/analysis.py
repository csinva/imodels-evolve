import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('amtl.csv')

print("="*80)
print("DATASET EXPLORATION")
print("="*80)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nSummary statistics:")
print(df.describe())
print(f"\nGenus distribution:")
print(df['genus'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum())

# Create a binary variable for Homo sapiens vs non-human primates
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Calculate AMTL rate (proportion of teeth lost)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print("\n" + "="*80)
print("AMTL RATE BY GENUS")
print("="*80)
print(df.groupby('genus')['amtl_rate'].agg(['mean', 'std', 'count']))

# Statistical comparison of AMTL rates
human_amtl = df[df['genus'] == 'Homo sapiens']['amtl_rate']
non_human_amtl = df[df['genus'] != 'Homo sapiens']['amtl_rate']

print("\n" + "="*80)
print("BASIC T-TEST COMPARISON")
print("="*80)
print(f"Human AMTL rate: mean={human_amtl.mean():.4f}, std={human_amtl.std():.4f}")
print(f"Non-human AMTL rate: mean={non_human_amtl.mean():.4f}, std={non_human_amtl.std():.4f}")

# Perform t-test
t_stat, p_value_ttest = stats.ttest_ind(human_amtl, non_human_amtl)
print(f"\nT-test: t={t_stat:.4f}, p-value={p_value_ttest:.6f}")

# Now we need to account for age, sex, and tooth class
# Create dummy variables for categorical features
df_analysis = df.copy()
df_analysis = pd.get_dummies(df_analysis, columns=['tooth_class'], drop_first=False)

# Prepare features for regression
feature_cols = ['age', 'prob_male', 'is_human']
# Add tooth class dummies
tooth_class_cols = [col for col in df_analysis.columns if col.startswith('tooth_class_')]
feature_cols.extend(tooth_class_cols)

# Remove rows with missing values in key columns
df_clean = df_analysis[feature_cols + ['amtl_rate', 'num_amtl', 'sockets']].dropna()

print("\n" + "="*80)
print("REGRESSION ANALYSIS (Controlling for Age, Sex, Tooth Class)")
print("="*80)
print(f"Sample size for regression: {len(df_clean)}")

# Linear regression on AMTL rate
X = df_clean[feature_cols]
y = df_clean['amtl_rate']

# Standardize features for interpretability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Add constant for statsmodels
X_with_const = sm.add_constant(X_scaled_df)

# Fit OLS model
model_ols = sm.OLS(y, X_with_const).fit()
print("\nLinear Regression Results (OLS):")
print(model_ols.summary())

# Extract key results for is_human
is_human_coef = model_ols.params['is_human']
is_human_pvalue = model_ols.pvalues['is_human']
is_human_ci = model_ols.conf_int().loc['is_human']

print("\n" + "="*80)
print("KEY FINDING: Effect of Being Human (Homo sapiens)")
print("="*80)
print(f"Coefficient: {is_human_coef:.6f}")
print(f"P-value: {is_human_pvalue:.6f}")
print(f"95% Confidence Interval: [{is_human_ci[0]:.6f}, {is_human_ci[1]:.6f}]")
print(f"Significant at 0.05 level: {is_human_pvalue < 0.05}")

# Additional analysis: Logistic regression using binomial outcome
# Create a weighted dataset for binomial regression
# For each row, we have num_amtl successes out of sockets trials

print("\n" + "="*80)
print("BINOMIAL LOGISTIC REGRESSION")
print("="*80)

# Expand the data for binomial logistic regression
# Create binary outcome (1 for each lost tooth, 0 for each retained tooth)
rows_expanded = []
for idx, row in df_clean.iterrows():
    for i in range(int(row['sockets'])):
        new_row = row[feature_cols].to_dict()
        if i < row['num_amtl']:
            new_row['tooth_lost'] = 1
        else:
            new_row['tooth_lost'] = 0
        rows_expanded.append(new_row)

df_expanded = pd.DataFrame(rows_expanded)

# Fit logistic regression
X_logistic = df_expanded[feature_cols]
y_logistic = df_expanded['tooth_lost']

# Standardize
X_logistic_scaled = scaler.fit_transform(X_logistic)
X_logistic_scaled_df = pd.DataFrame(X_logistic_scaled, columns=feature_cols)

# Use statsmodels for logistic regression with p-values
X_logistic_const = sm.add_constant(X_logistic_scaled_df)
logit_model = sm.Logit(y_logistic, X_logistic_const).fit(disp=0)

print("\nLogistic Regression Results:")
print(logit_model.summary())

# Extract key results for is_human
is_human_logit_coef = logit_model.params['is_human']
is_human_logit_pvalue = logit_model.pvalues['is_human']
is_human_logit_or = np.exp(is_human_logit_coef)

print("\n" + "="*80)
print("KEY FINDING: Effect of Being Human (Logistic Regression)")
print("="*80)
print(f"Coefficient (log-odds): {is_human_logit_coef:.6f}")
print(f"Odds Ratio: {is_human_logit_or:.6f}")
print(f"P-value: {is_human_logit_pvalue:.6f}")
print(f"Significant at 0.05 level: {is_human_logit_pvalue < 0.05}")

# Interpretation
print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)

if is_human_logit_pvalue < 0.05:
    if is_human_logit_coef > 0:
        direction = "HIGHER"
        interpretation = f"After controlling for age, sex, and tooth class, modern humans (Homo sapiens) have significantly HIGHER frequencies of AMTL compared to non-human primates (p={is_human_logit_pvalue:.6f}). The odds of tooth loss are {is_human_logit_or:.2f} times higher for humans."
    else:
        direction = "LOWER"
        interpretation = f"After controlling for age, sex, and tooth class, modern humans (Homo sapiens) have significantly LOWER frequencies of AMTL compared to non-human primates (p={is_human_logit_pvalue:.6f}). The odds of tooth loss are {is_human_logit_or:.2f} times the odds for non-human primates."
else:
    direction = "NO SIGNIFICANT DIFFERENCE"
    interpretation = f"After controlling for age, sex, and tooth class, there is NO statistically significant difference in AMTL frequencies between modern humans and non-human primates (p={is_human_logit_pvalue:.6f})."

print(interpretation)

# Determine response score
# If significant and positive effect: high score (70-90)
# If significant and negative effect: low score (10-30)
# If not significant: moderate score leaning toward observed direction (40-60)

if is_human_logit_pvalue < 0.05:
    if is_human_logit_coef > 0:
        # Significant positive effect - humans have higher AMTL
        response_score = 80
    else:
        # Significant negative effect - humans have lower AMTL
        response_score = 20
else:
    # Not significant - look at effect size
    if is_human_logit_coef > 0:
        response_score = 50  # Slight trend toward higher but not significant
    else:
        response_score = 45  # Slight trend toward lower but not significant

explanation = f"Research Question: Do modern humans have higher AMTL frequencies than non-human primates after controlling for age, sex, and tooth class? " \
              f"Analysis: Using binomial logistic regression controlling for age (continuous), sex (prob_male), and tooth class (anterior/posterior/premolar), " \
              f"the coefficient for Homo sapiens is {is_human_logit_coef:.4f} (p={is_human_logit_pvalue:.6f}). " \
              f"{interpretation}"

print("\n" + "="*80)
print("FINAL CONCLUSION")
print("="*80)
print(f"Response Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "="*80)
print("Analysis complete! conclusion.txt has been created.")
print("="*80)
