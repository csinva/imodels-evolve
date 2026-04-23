import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from imodels import RuleFitClassifier

# Load the data
df = pd.read_csv('crofoot.csv')

print("=" * 80)
print("DATA EXPLORATION")
print("=" * 80)
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nSummary statistics:")
print(df.describe())

# Create key derived variables
# Relative group size: difference in group size (focal - other)
df['relative_size'] = df['n_focal'] - df['n_other']

# Contest location: we'll use distance metrics
# Location advantage: (dist_other - dist_focal) 
# Positive = closer to focal's home, negative = closer to other's home
df['location_advantage'] = df['dist_other'] - df['dist_focal']

# Also create a binary indicator: is focal closer to its home than other is?
df['focal_closer_to_home'] = (df['dist_focal'] < df['dist_other']).astype(int)

print("\n" + "=" * 80)
print("DERIVED VARIABLES")
print("=" * 80)
print(f"\nRelative size (n_focal - n_other):")
print(df['relative_size'].describe())
print(f"\nLocation advantage (dist_other - dist_focal):")
print(df['location_advantage'].describe())
print(f"\nWin rate by relative size:")
print(df.groupby('relative_size')['win'].agg(['mean', 'count']))

print("\n" + "=" * 80)
print("CORRELATIONS")
print("=" * 80)
corr_vars = ['win', 'relative_size', 'location_advantage', 'dist_focal', 'dist_other', 
             'n_focal', 'n_other']
print(df[corr_vars].corr()['win'].sort_values(ascending=False))

print("\n" + "=" * 80)
print("STATISTICAL TESTS")
print("=" * 80)

# Test 1: Does relative group size affect winning?
print("\n1. Correlation between relative group size and winning:")
corr_size, p_size = stats.pearsonr(df['relative_size'], df['win'])
print(f"   Pearson r = {corr_size:.4f}, p-value = {p_size:.4f}")

# Test 2: Does location advantage affect winning?
print("\n2. Correlation between location advantage and winning:")
corr_loc, p_loc = stats.pearsonr(df['location_advantage'], df['win'])
print(f"   Pearson r = {corr_loc:.4f}, p-value = {p_loc:.4f}")

# Test 3: T-test - do winners have different relative sizes?
winners = df[df['win'] == 1]['relative_size']
losers = df[df['win'] == 0]['relative_size']
t_stat_size, p_ttest_size = stats.ttest_ind(winners, losers)
print(f"\n3. T-test for relative size (winners vs losers):")
print(f"   Winners mean relative size: {winners.mean():.2f}")
print(f"   Losers mean relative size: {losers.mean():.2f}")
print(f"   t-statistic = {t_stat_size:.4f}, p-value = {p_ttest_size:.4f}")

# Test 4: T-test - do winners have different location advantages?
winners_loc = df[df['win'] == 1]['location_advantage']
losers_loc = df[df['win'] == 0]['location_advantage']
t_stat_loc, p_ttest_loc = stats.ttest_ind(winners_loc, losers_loc)
print(f"\n4. T-test for location advantage (winners vs losers):")
print(f"   Winners mean location advantage: {winners_loc.mean():.2f}")
print(f"   Losers mean location advantage: {losers_loc.mean():.2f}")
print(f"   t-statistic = {t_stat_loc:.4f}, p-value = {p_ttest_loc:.4f}")

print("\n" + "=" * 80)
print("LOGISTIC REGRESSION MODEL")
print("=" * 80)

# Build logistic regression model with statsmodels for p-values
X = df[['relative_size', 'location_advantage']].copy()
X = sm.add_constant(X)
y = df['win']

logit_model = sm.Logit(y, X).fit(disp=False)
print(logit_model.summary())

print("\n" + "=" * 80)
print("INTERPRETABLE MODEL: RULEFIT")
print("=" * 80)

# Use RuleFit for interpretable rules
X_features = df[['relative_size', 'location_advantage', 'n_focal', 'n_other', 
                  'dist_focal', 'dist_other']].values
y = df['win'].values

try:
    rulefit = RuleFitClassifier(max_rules=10, random_state=42)
    rulefit.fit(X_features, y)
    
    print(f"\nRuleFit accuracy: {rulefit.score(X_features, y):.3f}")
    
    # Get rules
    rules_df = rulefit.get_rules()
    if rules_df is not None and len(rules_df) > 0:
        print("\nTop rules:")
        print(rules_df.head(10))
except Exception as e:
    print(f"RuleFit error: {e}")
    print("Continuing with other analyses...")

print("\n" + "=" * 80)
print("SCIKIT-LEARN LOGISTIC REGRESSION")
print("=" * 80)

# Standard sklearn logistic regression
X_std = df[['relative_size', 'location_advantage']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_std)

lr = LogisticRegression(random_state=42)
lr.fit(X_scaled, y)

print(f"\nLogistic Regression Coefficients (standardized):")
print(f"  Relative size: {lr.coef_[0][0]:.4f}")
print(f"  Location advantage: {lr.coef_[0][1]:.4f}")
print(f"  Intercept: {lr.intercept_[0]:.4f}")
print(f"\nModel accuracy: {lr.score(X_scaled, y):.3f}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

# Analyze results
significant_findings = []
explanation_parts = []

# Check relative group size
if p_size < 0.05:
    significant_findings.append("relative_size")
    effect_dir = "larger" if corr_size > 0 else "smaller"
    explanation_parts.append(f"Relative group size significantly affects winning (r={corr_size:.3f}, p={p_size:.4f}): {effect_dir} focal groups are more likely to win")

if p_ttest_size < 0.05:
    if "relative_size" not in significant_findings:
        significant_findings.append("relative_size")
    diff = winners.mean() - losers.mean()
    explanation_parts.append(f"Winners have significantly different relative group sizes (Δ={diff:.2f}, p={p_ttest_size:.4f})")

# Check location
if p_loc < 0.05:
    significant_findings.append("location")
    effect_dir = "closer to home" if corr_loc > 0 else "farther from home"
    explanation_parts.append(f"Contest location significantly affects winning (r={corr_loc:.3f}, p={p_loc:.4f}): being {effect_dir} increases win probability")

if p_ttest_loc < 0.05:
    if "location" not in significant_findings:
        significant_findings.append("location")
    diff = winners_loc.mean() - losers_loc.mean()
    explanation_parts.append(f"Winners have significantly different location advantages (Δ={diff:.2f}m, p={p_ttest_loc:.4f})")

# Check logistic regression coefficients
logit_pvals = logit_model.pvalues
if logit_pvals['relative_size'] < 0.05:
    coef = logit_model.params['relative_size']
    if "relative_size" not in significant_findings:
        significant_findings.append("relative_size")
    explanation_parts.append(f"Logistic regression confirms relative size effect (β={coef:.3f}, p={logit_pvals['relative_size']:.4f})")

if logit_pvals['location_advantage'] < 0.05:
    coef = logit_model.params['location_advantage']
    if "location" not in significant_findings:
        significant_findings.append("location")
    explanation_parts.append(f"Logistic regression confirms location effect (β={coef:.4f}, p={logit_pvals['location_advantage']:.4f})")

# Determine response score
print(f"\nSignificant factors found: {significant_findings}")
print(f"\nDetailed findings:")
for finding in explanation_parts:
    print(f"  - {finding}")

# Calculate response score
# The question asks how BOTH relative group size AND location influence winning
# We need both to be significant for a strong "Yes"

if len(significant_findings) == 0:
    response_score = 5
    explanation = "No significant effects found for either relative group size or contest location on winning probability."
elif 'relative_size' in significant_findings and 'location' in significant_findings:
    # Both factors are significant - strong Yes
    response_score = 90
    explanation = f"Both relative group size and contest location significantly influence winning. {' '.join(explanation_parts)}"
elif len(significant_findings) == 1:
    # Only one factor is significant - moderate Yes
    if 'relative_size' in significant_findings:
        response_score = 60
        explanation = f"Relative group size significantly influences winning, but contest location does not show significant effects. {' '.join(explanation_parts)}"
    else:
        response_score = 60
        explanation = f"Contest location significantly influences winning, but relative group size does not show significant effects. {' '.join(explanation_parts)}"
else:
    # Mixed or weak evidence
    response_score = 50
    explanation = f"Mixed evidence for effects. {' '.join(explanation_parts)}"

print(f"\n{'=' * 80}")
print(f"FINAL ANSWER")
print(f"{'=' * 80}")
print(f"Response score: {response_score}")
print(f"Explanation: {explanation}")

# Write conclusion to file
output = {
    "response": response_score,
    "explanation": explanation
}

with open('conclusion.txt', 'w') as f:
    json.dump(output, f)

print(f"\n✓ Written conclusion to conclusion.txt")
