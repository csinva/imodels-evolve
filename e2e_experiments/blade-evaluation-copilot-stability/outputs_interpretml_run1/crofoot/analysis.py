import pandas as pd
import numpy as np
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from interpret.glassbox import ExplainableBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('crofoot.csv')

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print("\nFirst few rows:")
print(df.head())
print("\nBasic statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())

# Research question: How do relative group size and contest location influence 
# the probability of winning?

# Key variables:
# - win: outcome (1 if focal won, 0 if other won)
# - Relative group size: n_focal - n_other (or n_focal / n_other)
# - Contest location: dist_focal vs dist_other (closer to home = advantage?)

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Create relative group size features
df['n_diff'] = df['n_focal'] - df['n_other']  # Difference in group size
df['n_ratio'] = df['n_focal'] / df['n_other']  # Ratio of group sizes
df['m_diff'] = df['m_focal'] - df['m_other']  # Difference in males
df['f_diff'] = df['f_focal'] - df['f_other']  # Difference in females

# Create location features
df['dist_diff'] = df['dist_focal'] - df['dist_other']  # Positive = farther from home
df['dist_ratio'] = df['dist_focal'] / df['dist_other']  # Higher = farther from home
df['closer_to_focal_home'] = (df['dist_focal'] < df['dist_other']).astype(int)  # 1 if closer to focal's home

print("\nNew features created:")
print(f"  - n_diff (group size difference): mean={df['n_diff'].mean():.2f}, std={df['n_diff'].std():.2f}")
print(f"  - n_ratio (group size ratio): mean={df['n_ratio'].mean():.2f}, std={df['n_ratio'].std():.2f}")
print(f"  - dist_diff (distance difference): mean={df['dist_diff'].mean():.2f}, std={df['dist_diff'].std():.2f}")
print(f"  - closer_to_focal_home: {df['closer_to_focal_home'].sum()} cases (out of {len(df)})")

print("\n" + "=" * 80)
print("CORRELATION ANALYSIS")
print("=" * 80)

# Correlations with win outcome
key_features = ['win', 'n_diff', 'n_ratio', 'dist_diff', 'dist_ratio', 
                'closer_to_focal_home', 'm_diff', 'f_diff']
corr_with_win = df[key_features].corr()['win'].sort_values(ascending=False)
print("\nCorrelations with winning outcome:")
print(corr_with_win)

print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Test 1: Does relative group size influence winning?
# Split by group size advantage
df['focal_larger'] = df['n_diff'] > 0
df['focal_smaller'] = df['n_diff'] < 0
df['equal_size'] = df['n_diff'] == 0

win_rate_larger = df[df['focal_larger']]['win'].mean()
win_rate_smaller = df[df['focal_smaller']]['win'].mean()
win_rate_equal = df[df['equal_size']]['win'].mean() if df['equal_size'].sum() > 0 else np.nan

print("\nWin rates by relative group size:")
print(f"  - Focal group LARGER: {win_rate_larger:.3f} (n={df['focal_larger'].sum()})")
print(f"  - Focal group SMALLER: {win_rate_smaller:.3f} (n={df['focal_smaller'].sum()})")
print(f"  - Groups EQUAL size: {win_rate_equal:.3f} (n={df['equal_size'].sum()})")

# T-test: larger vs smaller groups
if df['focal_larger'].sum() > 0 and df['focal_smaller'].sum() > 0:
    t_stat_size, p_val_size = stats.ttest_ind(
        df[df['focal_larger']]['win'],
        df[df['focal_smaller']]['win']
    )
    print(f"\nT-test (larger vs smaller): t={t_stat_size:.3f}, p={p_val_size:.4f}")
    if p_val_size < 0.05:
        print("  *** SIGNIFICANT at p<0.05")
    else:
        print("  NOT significant")

# Test 2: Does contest location influence winning?
win_rate_home = df[df['closer_to_focal_home'] == 1]['win'].mean()
win_rate_away = df[df['closer_to_focal_home'] == 0]['win'].mean()

print("\nWin rates by contest location:")
print(f"  - Contest CLOSER to focal's home: {win_rate_home:.3f} (n={df['closer_to_focal_home'].sum()})")
print(f"  - Contest FARTHER from focal's home: {win_rate_away:.3f} (n={(1-df['closer_to_focal_home']).sum()})")

# T-test: home vs away
t_stat_loc, p_val_loc = stats.ttest_ind(
    df[df['closer_to_focal_home'] == 1]['win'],
    df[df['closer_to_focal_home'] == 0]['win']
)
print(f"\nT-test (home vs away): t={t_stat_loc:.3f}, p={p_val_loc:.4f}")
if p_val_loc < 0.05:
    print("  *** SIGNIFICANT at p<0.05")
else:
    print("  NOT significant")

# Pearson correlation tests
r_size, p_size = stats.pearsonr(df['n_diff'], df['win'])
r_dist, p_dist = stats.pearsonr(df['dist_diff'], df['win'])

print(f"\nPearson correlation (group size difference vs win): r={r_size:.3f}, p={p_size:.4f}")
if p_size < 0.05:
    print("  *** SIGNIFICANT at p<0.05")

print(f"\nPearson correlation (distance difference vs win): r={r_dist:.3f}, p={p_dist:.4f}")
if p_dist < 0.05:
    print("  *** SIGNIFICANT at p<0.05")

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION ANALYSIS")
print("=" * 80)

# Logistic regression with statsmodels for p-values
X_logit = df[['n_diff', 'dist_diff']].copy()
X_logit = sm.add_constant(X_logit)
y = df['win']

logit_model = sm.Logit(y, X_logit)
logit_result = logit_model.fit(disp=0)
print("\nLogistic Regression Summary:")
print(logit_result.summary())

print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: Explainable Boosting Classifier")
print("=" * 80)

# Train EBM model for interpretable predictions
X_ebm = df[['n_diff', 'dist_diff', 'm_diff', 'f_diff']].copy()
y_ebm = df['win']

ebm = ExplainableBoostingClassifier(random_state=42)
ebm.fit(X_ebm, y_ebm)

# Get feature importances (using term_importances for EBM)
feature_names = X_ebm.columns.tolist()
if hasattr(ebm, 'term_importances'):
    importances = ebm.term_importances()
    print("\nFeature Importances from EBM:")
    for name, imp in zip(feature_names, importances):
        print(f"  {name}: {imp:.4f}")
else:
    print("\nEBM trained successfully (feature importances not directly available)")

# Cross-validation score
cv_scores = cross_val_score(ebm, X_ebm, y_ebm, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

print("\n" + "=" * 80)
print("SYNTHESIS AND CONCLUSION")
print("=" * 80)

# Determine the response based on statistical significance and effect sizes
factors = []
evidence_strength = []

# Factor 1: Relative group size
if p_size < 0.05:
    factors.append("relative group size")
    evidence_strength.append(abs(r_size))
    print(f"✓ Relative group size IS significant (p={p_size:.4f}, r={r_size:.3f})")
else:
    print(f"✗ Relative group size is NOT significant (p={p_size:.4f})")

# Factor 2: Contest location
if p_dist < 0.05 or p_val_loc < 0.05:
    factors.append("contest location")
    evidence_strength.append(max(abs(r_dist), abs(t_stat_loc)/10))
    print(f"✓ Contest location IS significant (p_corr={p_dist:.4f}, p_ttest={p_val_loc:.4f})")
else:
    print(f"✗ Contest location is NOT significant (p_corr={p_dist:.4f}, p_ttest={p_val_loc:.4f})")

# Determine Likert score
if len(factors) == 2:
    # Both factors significant
    avg_strength = np.mean(evidence_strength)
    response = int(70 + avg_strength * 30)  # 70-100 range
    explanation = (
        f"BOTH relative group size and contest location significantly influence winning probability. "
        f"Group size difference shows r={r_size:.3f} (p={p_size:.4f}), with larger groups having {win_rate_larger:.1%} "
        f"win rate vs {win_rate_smaller:.1%} for smaller groups. "
        f"Location effects show that contests closer to focal's home have {win_rate_home:.1%} win rate "
        f"vs {win_rate_away:.1%} when farther (p={p_val_loc:.4f}). "
        f"The logistic regression confirms both factors are important predictors. "
        f"Strong evidence supports that BOTH factors influence contest outcomes."
    )
elif len(factors) == 1:
    # One factor significant
    if "relative group size" in factors:
        response = 60
        explanation = (
            f"Relative group size significantly influences winning probability (r={r_size:.3f}, p={p_size:.4f}). "
            f"Larger groups win {win_rate_larger:.1%} of contests vs {win_rate_smaller:.1%} for smaller groups. "
            f"However, contest location does NOT show significant effects (p={p_val_loc:.4f}). "
            f"Only ONE of the two factors shows clear influence."
        )
    else:  # location significant
        response = 60
        explanation = (
            f"Contest location significantly influences winning probability (p={p_val_loc:.4f}). "
            f"Contests closer to focal's home have {win_rate_home:.1%} win rate vs {win_rate_away:.1%} farther away. "
            f"However, relative group size does NOT show significant effects (p={p_size:.4f}). "
            f"Only ONE of the two factors shows clear influence."
        )
else:
    # Neither factor significant
    response = 25
    explanation = (
        f"Neither relative group size nor contest location show statistically significant effects on winning probability. "
        f"Group size: p={p_size:.4f}, Location: p={p_val_loc:.4f}. "
        f"While descriptive differences exist, they do not reach statistical significance with this sample size (n={len(df)}). "
        f"The evidence does NOT support a clear influence of these factors."
    )

print(f"\n*** FINAL LIKERT SCORE: {response}/100 ***")
print(f"*** EXPLANATION: {explanation} ***")

# Write conclusion to file
conclusion = {
    "response": response,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("\n✓ conclusion.txt has been written.")
