import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import json

# Load the data
df = pd.read_csv('crofoot.csv')

print("=" * 80)
print("DATASET OVERVIEW")
print("=" * 80)
print(f"Shape: {df.shape}")
print("\nFirst few rows:")
print(df.head())
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Research question: How do relative group size and contest location influence 
# the probability of winning?

# Create derived features for analysis
df['relative_size'] = df['n_focal'] - df['n_other']  # Positive = focal is larger
df['relative_males'] = df['m_focal'] - df['m_other']
df['relative_females'] = df['f_focal'] - df['f_other']
df['home_advantage'] = df['dist_other'] - df['dist_focal']  # Positive = closer to focal's home

print("\n" + "=" * 80)
print("DERIVED FEATURES")
print("=" * 80)
print("\nRelative size (n_focal - n_other):")
print(df['relative_size'].describe())
print("\nHome advantage (dist_other - dist_focal):")
print(df['home_advantage'].describe())

print("\n" + "=" * 80)
print("EXPLORATORY ANALYSIS")
print("=" * 80)

# Win rate overall
win_rate = df['win'].mean()
print(f"\nOverall win rate for focal group: {win_rate:.3f}")

# Win rate by relative size
print("\nWin rate by relative group size:")
for rel_size in sorted(df['relative_size'].unique()):
    subset = df[df['relative_size'] == rel_size]
    win_rate_subset = subset['win'].mean()
    print(f"  Relative size {rel_size:+3d}: {win_rate_subset:.3f} (n={len(subset)})")

# Correlation analysis
print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)
features_of_interest = ['relative_size', 'home_advantage', 'relative_males', 
                        'relative_females', 'dist_focal', 'dist_other']
correlations = df[features_of_interest + ['win']].corr()['win'].drop('win')
print("\nCorrelations with winning:")
print(correlations.sort_values(ascending=False))

# Statistical tests
print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Test 1: Does relative group size affect winning?
larger_groups = df[df['relative_size'] > 0]['win']
smaller_groups = df[df['relative_size'] < 0]['win']
equal_groups = df[df['relative_size'] == 0]['win']

if len(larger_groups) > 0 and len(smaller_groups) > 0:
    t_stat_size, p_val_size = stats.ttest_ind(larger_groups, smaller_groups)
    print(f"\nT-test: Larger vs Smaller groups")
    print(f"  Larger groups win rate: {larger_groups.mean():.3f} (n={len(larger_groups)})")
    print(f"  Smaller groups win rate: {smaller_groups.mean():.3f} (n={len(smaller_groups)})")
    print(f"  t-statistic: {t_stat_size:.3f}, p-value: {p_val_size:.4f}")

# Test 2: Does location (home advantage) affect winning?
median_home_adv = df['home_advantage'].median()
close_to_home = df[df['home_advantage'] > median_home_adv]['win']
far_from_home = df[df['home_advantage'] <= median_home_adv]['win']

t_stat_loc, p_val_loc = stats.ttest_ind(close_to_home, far_from_home)
print(f"\nT-test: Close to focal's home vs Far from focal's home")
print(f"  Close to home win rate: {close_to_home.mean():.3f} (n={len(close_to_home)})")
print(f"  Far from home win rate: {far_from_home.mean():.3f} (n={len(far_from_home)})")
print(f"  t-statistic: {t_stat_loc:.3f}, p-value: {p_val_loc:.4f}")

# Correlation tests
r_size, p_size_corr = stats.pearsonr(df['relative_size'], df['win'])
r_home, p_home_corr = stats.pearsonr(df['home_advantage'], df['win'])

print(f"\nPearson correlation tests:")
print(f"  Relative size vs win: r={r_size:.3f}, p={p_size_corr:.4f}")
print(f"  Home advantage vs win: r={r_home:.3f}, p={p_home_corr:.4f}")

# Logistic Regression with statsmodels for p-values
print("\n" + "=" * 80)
print("LOGISTIC REGRESSION (statsmodels)")
print("=" * 80)

X = df[['relative_size', 'home_advantage']].copy()
X = sm.add_constant(X)
y = df['win']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)
print("\nLogistic Regression Summary:")
print(result.summary2())

# Interpretable sklearn model
print("\n" + "=" * 80)
print("INTERPRETABLE SKLEARN MODEL")
print("=" * 80)

X_sklearn = df[['relative_size', 'home_advantage']].values
y_sklearn = df['win'].values

# Standardize for coefficient interpretation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sklearn)

log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_scaled, y_sklearn)

print("\nLogistic Regression Coefficients (standardized):")
print(f"  Relative size: {log_reg.coef_[0][0]:.4f}")
print(f"  Home advantage: {log_reg.coef_[0][1]:.4f}")
print(f"  Intercept: {log_reg.intercept_[0]:.4f}")

# Decision tree for interpretability
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_sklearn, y_sklearn)
print("\nDecision Tree Feature Importances:")
print(f"  Relative size: {dt.feature_importances_[0]:.4f}")
print(f"  Home advantage: {dt.feature_importances_[1]:.4f}")

# Try imodels if available
try:
    from imodels import HSTreeClassifier
    
    hst = HSTreeClassifier(random_state=42)
    hst.fit(X_sklearn, y_sklearn)
    print("\nHSTree Feature Importances:")
    if hasattr(hst, 'feature_importances_'):
        print(f"  Relative size: {hst.feature_importances_[0]:.4f}")
        print(f"  Home advantage: {hst.feature_importances_[1]:.4f}")
except Exception as e:
    print(f"\nimodels error: {e}, skipping")

# ANOVA for categorical analysis
print("\n" + "=" * 80)
print("ANOVA ANALYSIS")
print("=" * 80)

# Create size categories
df['size_category'] = pd.cut(df['relative_size'], bins=[-np.inf, -1, 1, np.inf], 
                               labels=['smaller', 'equal', 'larger'])

# ANOVA for relative size
groups_size = [group['win'].values for name, group in df.groupby('size_category')]
f_stat_size, p_val_anova_size = stats.f_oneway(*groups_size)
print(f"\nANOVA: Effect of size category on winning")
print(f"  F-statistic: {f_stat_size:.3f}, p-value: {p_val_anova_size:.4f}")

# Create location categories
df['location_category'] = pd.cut(df['home_advantage'], bins=[-np.inf, 0, np.inf], 
                                  labels=['away', 'home'])
groups_loc = [group['win'].values for name, group in df.groupby('location_category')]
f_stat_loc, p_val_anova_loc = stats.f_oneway(*groups_loc)
print(f"\nANOVA: Effect of location on winning")
print(f"  F-statistic: {f_stat_loc:.3f}, p-value: {p_val_anova_loc:.4f}")

# CONCLUSION
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Evaluate evidence for both factors
evidence_summary = {
    'relative_size': {
        'correlation': (r_size, p_size_corr),
        'logit_pvalue': result.pvalues['relative_size'],
        'coefficient_sign': result.params['relative_size'] > 0
    },
    'home_advantage': {
        'correlation': (r_home, p_home_corr),
        'logit_pvalue': result.pvalues['home_advantage'],
        'coefficient_sign': result.params['home_advantage'] > 0
    }
}

print("\nEvidence summary:")
print(f"Relative size: p={p_size_corr:.4f} (correlation), p={result.pvalues['relative_size']:.4f} (logit)")
print(f"Home advantage: p={p_home_corr:.4f} (correlation), p={result.pvalues['home_advantage']:.4f} (logit)")

# Determine response based on statistical significance
# Using alpha = 0.05 as threshold
significant_size = (p_size_corr < 0.05) or (result.pvalues['relative_size'] < 0.05)
significant_location = (p_home_corr < 0.05) or (result.pvalues['home_advantage'] < 0.05)

# Both factors need to show influence for a strong "Yes"
if significant_size and significant_location:
    response_score = 85
    explanation = (
        f"Both relative group size and contest location significantly influence winning probability. "
        f"Relative size shows correlation r={r_size:.3f} (p={p_size_corr:.4f}) and logistic regression "
        f"coefficient p={result.pvalues['relative_size']:.4f}. Home advantage shows correlation "
        f"r={r_home:.3f} (p={p_home_corr:.4f}) and logistic regression coefficient "
        f"p={result.pvalues['home_advantage']:.4f}. Both effects are statistically significant (p<0.05)."
    )
elif significant_size or significant_location:
    response_score = 60
    if significant_size:
        explanation = (
            f"Relative group size significantly influences winning (p={p_size_corr:.4f}), "
            f"but contest location shows weaker evidence (p={p_home_corr:.4f}). "
            f"Only partial support for both factors."
        )
    else:
        explanation = (
            f"Contest location significantly influences winning (p={p_home_corr:.4f}), "
            f"but relative group size shows weaker evidence (p={p_size_corr:.4f}). "
            f"Only partial support for both factors."
        )
elif (p_size_corr < 0.10 and p_home_corr < 0.10):
    response_score = 45
    explanation = (
        f"Both factors show trends but lack strong statistical significance. "
        f"Relative size p={p_size_corr:.4f}, location p={p_home_corr:.4f}. "
        f"Marginal evidence for influence of both factors."
    )
else:
    response_score = 25
    explanation = (
        f"Neither relative group size nor contest location show strong statistical significance. "
        f"Relative size p={p_size_corr:.4f}, location p={p_home_corr:.4f}. "
        f"Limited evidence for systematic influence."
    )

print(f"\nResponse score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
conclusion = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n" + "=" * 80)
print("Analysis complete. Results written to conclusion.txt")
print("=" * 80)
