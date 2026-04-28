import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('amtl.csv')

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\n\nGenus distribution:")
print(df['genus'].value_counts())

print(f"\n\nBasic statistics:")
print(df.describe())

# Calculate AMTL rate (proportion of missing teeth)
df['amtl_rate'] = df['num_amtl'] / df['sockets']

print(f"\n\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].agg(['mean', 'std', 'count']))

# Create binary indicator for Homo sapiens
df['is_homo_sapiens'] = (df['genus'] == 'Homo sapiens').astype(int)

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS")
print("=" * 80)

# 1. Simple comparison: Homo sapiens vs all non-human primates
homo_amtl = df[df['genus'] == 'Homo sapiens']['amtl_rate']
non_homo_amtl = df[df['genus'] != 'Homo sapiens']['amtl_rate']

t_stat, p_value = stats.ttest_ind(homo_amtl, non_homo_amtl)
print(f"\n\nT-test: Homo sapiens vs Non-human primates")
print(f"Homo sapiens mean AMTL rate: {homo_amtl.mean():.4f}")
print(f"Non-human primates mean AMTL rate: {non_homo_amtl.mean():.4f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.6f}")

# 2. Mann-Whitney U test (non-parametric alternative)
u_stat, p_value_mw = stats.mannwhitneyu(homo_amtl, non_homo_amtl, alternative='two-sided')
print(f"\n\nMann-Whitney U test:")
print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value_mw:.6f}")

# 3. Regression analysis accounting for confounders
print("\n\n" + "=" * 80)
print("REGRESSION ANALYSIS (Accounting for Age, Sex, Tooth Class)")
print("=" * 80)

# Create dummy variables for tooth class
df_model = pd.get_dummies(df, columns=['tooth_class'], drop_first=True)

# Prepare data for logistic regression on individual teeth
# We need to expand the data so each tooth is a separate observation
rows = []
for idx, row in df.iterrows():
    num_lost = int(row['num_amtl'])
    num_present = int(row['sockets']) - num_lost
    
    # Add rows for lost teeth
    for _ in range(num_lost):
        rows.append({
            'lost': 1,
            'age': row['age'],
            'prob_male': row['prob_male'],
            'is_homo_sapiens': row['is_homo_sapiens'],
            'tooth_class_Posterior': row.get('tooth_class_Posterior', 0),
            'tooth_class_Premolar': row.get('tooth_class_Premolar', 0)
        })
    
    # Add rows for present teeth
    for _ in range(num_present):
        rows.append({
            'lost': 0,
            'age': row['age'],
            'prob_male': row['prob_male'],
            'is_homo_sapiens': row['is_homo_sapiens'],
            'tooth_class_Posterior': row.get('tooth_class_Posterior', 0),
            'tooth_class_Premolar': row.get('tooth_class_Premolar', 0)
        })

df_expanded = pd.DataFrame(rows)

print(f"\nExpanded dataset shape: {df_expanded.shape}")
print(f"Total teeth analyzed: {len(df_expanded)}")
print(f"Teeth lost: {df_expanded['lost'].sum()}")
print(f"Proportion lost: {df_expanded['lost'].mean():.4f}")

# Logistic regression with statsmodels for p-values
# Use sklearn's LogisticRegression first, then get detailed stats with a simpler model
from sklearn.linear_model import LogisticRegression as LR

X = df_expanded[['age', 'prob_male', 'is_homo_sapiens', 'tooth_class_Posterior', 'tooth_class_Premolar']]
y = df_expanded['lost']

# Fit with sklearn first
lr_model = LR(max_iter=1000, solver='lbfgs')
lr_model.fit(X, y)

print("\n\nLogistic Regression Coefficients (sklearn):")
for i, col in enumerate(X.columns):
    print(f"{col}: {lr_model.coef_[0][i]:.4f}")

# Now use statsmodels with better numerical stability
X_scaled = (X - X.mean()) / X.std()
X_scaled_const = sm.add_constant(X_scaled)

try:
    logit_model = sm.Logit(y, X_scaled_const)
    result = logit_model.fit(method='bfgs', maxiter=1000, disp=0)
    
    print("\n\nLogistic Regression Results (statsmodels - scaled):")
    print(result.summary())
    
    # Get the position of is_homo_sapiens in the scaled version
    homo_idx = list(X.columns).index('is_homo_sapiens')
    coef_homo_scaled = result.params[homo_idx + 1]  # +1 for constant
    pval_homo = result.pvalues[homo_idx + 1]
    
    # Convert back to original scale
    coef_homo = coef_homo_scaled / X['is_homo_sapiens'].std()
    
    # Get confidence interval
    conf_int = result.conf_int()
    ci_lower_scaled = conf_int.iloc[homo_idx + 1, 0]
    ci_upper_scaled = conf_int.iloc[homo_idx + 1, 1]
    ci_lower = ci_lower_scaled / X['is_homo_sapiens'].std()
    ci_upper = ci_upper_scaled / X['is_homo_sapiens'].std()
    
except Exception as e:
    print(f"\n\nStatsmodels fitting failed: {e}")
    print("Using sklearn coefficients and bootstrap for p-values")
    
    # Use sklearn coefficient
    homo_idx = list(X.columns).index('is_homo_sapiens')
    coef_homo = lr_model.coef_[0][homo_idx]
    
    # Bootstrap for confidence interval
    from sklearn.utils import resample
    n_bootstrap = 1000
    coef_bootstrap = []
    
    for _ in range(n_bootstrap):
        X_boot, y_boot = resample(X, y, random_state=_)
        lr_boot = LR(max_iter=1000, solver='lbfgs')
        lr_boot.fit(X_boot, y_boot)
        coef_bootstrap.append(lr_boot.coef_[0][homo_idx])
    
    # Calculate p-value (two-tailed test)
    coef_bootstrap = np.array(coef_bootstrap)
    pval_homo = 2 * min(np.mean(coef_bootstrap <= 0), np.mean(coef_bootstrap >= 0))
    
    # 95% confidence interval
    ci_lower = np.percentile(coef_bootstrap, 2.5)
    ci_upper = np.percentile(coef_bootstrap, 97.5)
    
    print(f"Bootstrap completed with {n_bootstrap} iterations")

print("\n\nLogistic Regression Results:")
print(f"Coefficient for is_homo_sapiens: {coef_homo:.4f}")
print(f"P-value: {pval_homo:.6f}")

print(f"\n\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

print(f"\n\nCoefficient for Homo sapiens: {coef_homo:.4f}")
print(f"P-value: {pval_homo:.6f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Calculate odds ratio
odds_ratio = np.exp(coef_homo)
print(f"\n\nOdds Ratio (Homo sapiens vs non-human primates): {odds_ratio:.4f}")
print(f"Interpretation: Homo sapiens have {odds_ratio:.2f}x the odds of AMTL compared to non-human primates,")
print(f"after controlling for age, sex, and tooth class.")

# 4. Additional analysis: Effect of age on AMTL by genus
print("\n\n" + "=" * 80)
print("AGE EFFECT ANALYSIS")
print("=" * 80)

correlation_by_genus = df.groupby('genus').apply(
    lambda x: stats.pearsonr(x['age'], x['amtl_rate'])
)

print("\n\nCorrelation between age and AMTL rate by genus:")
for genus in df['genus'].unique():
    genus_data = df[df['genus'] == genus]
    corr, pval = stats.pearsonr(genus_data['age'], genus_data['amtl_rate'])
    print(f"{genus}: r={corr:.4f}, p={pval:.6f}")

# Determine conclusion
print("\n\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Decision based on statistical significance and effect size
if pval_homo < 0.05:
    if coef_homo > 0:
        # Homo sapiens have SIGNIFICANTLY higher AMTL
        response_score = 95  # Strong Yes
        explanation = (
            f"After controlling for age, sex, and tooth class, Homo sapiens have significantly "
            f"higher AMTL rates (OR={odds_ratio:.2f}, p={pval_homo:.6f}). "
            f"The logistic regression coefficient is {coef_homo:.3f} with 95% CI [{ci_lower:.3f}, {ci_upper:.3f}], "
            f"indicating that modern humans have approximately {odds_ratio:.1f} times the odds of tooth loss "
            f"compared to non-human primates. This is statistically significant (p<0.05) and represents a substantial effect."
        )
    else:
        # Homo sapiens have SIGNIFICANTLY lower AMTL
        response_score = 5  # Strong No
        explanation = (
            f"After controlling for age, sex, and tooth class, Homo sapiens have significantly "
            f"LOWER AMTL rates (OR={odds_ratio:.2f}, p={pval_homo:.6f}). "
            f"The logistic regression coefficient is {coef_homo:.3f}, indicating that modern humans "
            f"have lower odds of tooth loss compared to non-human primates (p<0.05)."
        )
elif pval_homo < 0.10:
    # Marginally significant
    if coef_homo > 0:
        response_score = 65  # Moderate Yes
        explanation = (
            f"After controlling for age, sex, and tooth class, there is marginal evidence that "
            f"Homo sapiens have higher AMTL rates (OR={odds_ratio:.2f}, p={pval_homo:.3f}). "
            f"While the effect is positive, it is not conventionally significant (p={pval_homo:.3f})."
        )
    else:
        response_score = 35  # Moderate No
        explanation = (
            f"After controlling for age, sex, and tooth class, there is marginal evidence that "
            f"Homo sapiens have lower AMTL rates (OR={odds_ratio:.2f}, p={pval_homo:.3f})."
        )
else:
    # Not significant
    response_score = 50  # Neutral
    explanation = (
        f"After controlling for age, sex, and tooth class, there is no statistically significant "
        f"difference in AMTL rates between Homo sapiens and non-human primates (p={pval_homo:.3f}). "
        f"The odds ratio is {odds_ratio:.2f}, but this difference is not statistically significant."
    )

print(f"\n\nFinal Assessment:")
print(f"Response Score: {response_score}/100")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
